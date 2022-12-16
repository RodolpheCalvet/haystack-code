import os

from haystack.document_stores import ElasticsearchDocumentStore
from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http

host = os.environ.get("ELASTICSEARCH_HOST", "localhost")
print(f'host : {host}')

document_store = ElasticsearchDocumentStore(host='elasticsearch', username="", password="", index="document")
doc_dir = "C:/dataHaystack/shakespeare"
mod_dir = "C:/dataHaystack/mods_shakespeare"
wikiWill = "https://github.com/RodolpheCalvet/haystack-code/raw/main/data/wiki.zip"
fetch_archive_from_http(url=wikiWill, output_dir=doc_dir)

docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

document_store.write_documents(docs)



