import os

from haystack.document_stores import ElasticsearchDocumentStore
from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http
import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.DEBUG)#WARNING)
logging.getLogger("haystack").setLevel(logging.DEBUG)#

host = os.environ.get("ELASTICSEARCH_HOST", "localhost")
print(f'host : {host}')

document_store = ElasticsearchDocumentStore(username="", password="", index="document")
doc_dir = "C:/dataHaystack/shakespeare"
mod_dir = "C:/dataHaystack/mods_shakespeare"
wikiWill = "https://github.com/RodolpheCalvet/haystack/raw/main/data/wiki.zip"#WilliWiki.pdf"#wiki.txt"
#fetch_archive_from_http(url=wikiWill, output_dir=doc_dir)

docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

document_store.delete_documents()
document_store.write_documents(docs)



