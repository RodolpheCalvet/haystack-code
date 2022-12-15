import os
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http
from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from haystack.nodes import TransformersReader
from haystack import Pipeline
from haystack.pipelines import ExtractiveQAPipeline
from pprint import pprint
from haystack.utils import print_answers

host = os.environ.get("ELASTICSEARCH_HOST", "localhost")
print(f'host : {host}')

document_store = ElasticsearchDocumentStore(host=host, username="", password="", index="document")
doc_dir = "C:/dataHaystack/tutorial1"
mod_dir = "C:/dataHaystack/mods_tutorial1"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt1.zip"
#fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

#docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
#print(docs[:3])

#document_store.write_documents(docs)

retriever = BM25Retriever(document_store=document_store)

# SAVED ?
reader = FARMReader(mod_dir, use_gpu=False)

# PAS SAVED ?
# reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
# reader = TransformersReader(model_name_or_path="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased", use_gpu=-1)
# reader.save(mod_dir)

#*******************************************PIPE
eQAP = ExtractiveQAPipeline(reader, retriever)
# oQAP = Pipeline().add_node(reader)

p=eQAP
#*******************************************LINE

#prediction1 = p.run(
#    query="Who is the father of Arya Stark?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
#)

prediction2 = p.run(query="Who created the Dothraki vocabulary?", params={"Reader": {"top_k": 5}})
prediction3 = p.run(query="Who is the sister of Sansa?", params={"Reader": {"top_k": 5}})

pprint(prediction2)

print_answers(prediction2, details="minimum")
print_answers(prediction3, details="all")


print("Cool!")