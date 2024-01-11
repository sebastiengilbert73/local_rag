# Cf. https://learn.activeloop.ai/courses/take/rag/multimedia/51320356-iterative-optimization-of-llamaindex-rag-pipeline-a-step-by-step-approach
import logging
import os
from llama_index.node_parser import SimpleNodeParser
from llama_index import SimpleDirectoryReader

from llama_index import VectorStoreIndex, ServiceContext, StorageContext

from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM
from transformers import BitsAndBytesConfig
import torch

__import__('pysqlite3')  # https://gist.github.com/defulmere/8b9695e415a44271061cc8e272f3c300
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from llama_index.vector_stores import ChromaVectorStore

from llama_index.evaluation import generate_question_context_pairs
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s \t%(message)s')

def messages_to_prompt(messages):  # CF. https://colab.research.google.com/drive/16Ygf2IyGNkb725ZqtRmFQjwWBuzFX_kl?usp=sharing#scrollTo=lMNaHDzPM68f
  prompt = ""
  for message in messages:
    if message.role == 'system':
      prompt += f"<|system|>\n{message.content}</s>\n"
    elif message.role == 'user':
      prompt += f"<|user|>\n{message.content}</s>\n"
    elif message.role == 'assistant':
      prompt += f"<|assistant|>\n{message.content}</s>\n"

  # ensure we start with a system prompt, insert blank if needed
  if not prompt.startswith("<|system|>\n"):
    prompt = "<|system|>\n</s>\n" + prompt

  # add final assistant prompt
  prompt = prompt + "<|assistant|>\n"

  return prompt

def main():
    logging.info("iterative_optimization_llamaindex_rag.main()")

    # Goal: No need for an API key!

    # First, we create Document LlamaIndex objects from the text data
    documents = SimpleDirectoryReader("./paul_graham").load_data()  # List of Document; len(documents) = 1
    node_parser = SimpleNodeParser.from_defaults(chunk_size=512)  # llama_index.node_parser.text.sentence.SentenceSplitter
    # node_parser.chunk_overlap = 200
    nodes = node_parser.get_nodes_from_documents(documents)  # List[llama_index.schema.TextNode]
    # len(nodes) = 56; # len(nodes[0].text.split()) = 355

    # By default, the node/chunks ids are set to random uuids.To ensure same id's per run, we manually set them.
    for idx, node in enumerate(nodes):
        node.id_ = f"node_{idx}"

    print(f"Number of Documents: {len(documents)}")
    print(f"Number of node: {len(nodes)} with the current chunk size of {node_parser.chunk_size}")

    # Cf. https://colab.research.google.com/drive/16Ygf2IyGNkb725ZqtRmFQjwWBuzFX_kl?usp=sharing
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    llm = HuggingFaceLLM(
        model_name="stabilityai/stablelm-zephyr-3b",  #"HuggingFaceH4/zephyr-7b-alpha", "stabilityai/stablelm-zephyr-3b"
        tokenizer_name="stabilityai/stablelm-zephyr-3b",  #"HuggingFaceH4/zephyr-7b-alpha", "stabilityai/stablelm-zephyr-3b"
        query_wrapper_prompt=PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
        context_window=2048, #3900,
        max_new_tokens=256,
        model_kwargs={"quantization_config": quantization_config, "trust_remote_code": True},
        # tokenizer_kwargs={},
        generate_kwargs={"do_sample": False},
        messages_to_prompt=messages_to_prompt,
        device_map="auto"
    )

    db_directory = "./chroma_db"
    embed_model = "local:BAAI/bge-small-en-v1.5"
    db_name = "paul_graham"

    db = chromadb.PersistentClient(path=db_directory)
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

    if db_name not in [c.name for c in db.list_collections()]:  # Create the index
        logging.info(f"Building the index...")
        chroma_collection = db.create_collection(db_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex(nodes, service_context=service_context, storage_context=storage_context,
                                        show_progress=True)
    else:  # The index was already created
        logging.info(f"Loading the db from {db_directory}...")
        chroma_collection = db.get_collection(db_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)



    # Build a query engine
    query_engine = vector_index.as_query_engine(similarity_top_k=5)
    logging.info("Before query_engine.query()")
    response_vector = query_engine.query("What are the main things Paul worked on before college?")
    logging.info("After query_engine.query()")
    print(response_vector.response)

    # Generate question-context pairs
    qc_dataset_filepath = "./qc_dataset.json"
    if not os.path.exists(qc_dataset_filepath):
        qc_dataset = generate_question_context_pairs(
            nodes,
            llm=llm,
            num_questions_per_chunk=1
        )
        # Save the dataset as a json file for later luse
        qc_dataset.save_json("qc_dataset.json")

    else:
        with open(qc_dataset_filepath, 'r') as json_file:
            qc_dataset = json.load(json_file)


if __name__ == '__main__':
    main()