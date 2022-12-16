import os

from nr.util.awsgi.launcher import gunicorn

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
doc_dir = "C:/dataHaystack/shakespeare"
mod_dir = "C:/dataHaystack/mods_shakespeare"
wikiWill = "https://github.com/RodolpheCalvet/haystack-code/raw/main/data/WilliWiki.zip"
fetch_archive_from_http(url=wikiWill, output_dir=doc_dir)

docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

document_store.write_documents(docs)

retriever = BM25Retriever(document_store=document_store)

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
# reader = TransformersReader(model_name_or_path="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased", use_gpu=-1)
reader.save(mod_dir)

eQAP = ExtractiveQAPipeline(reader, retriever)

p=eQAP


