from haystack.document_stores import FAISSDocumentStore
from haystack.utils import convert_files_to_docs, fetch_archive_from_http, clean_wiki_text
from haystack.nodes import DensePassageRetriever
from haystack.utils import print_documents
from haystack.pipelines import DocumentSearchPipeline
from haystack.nodes import Seq2SeqGenerator
from haystack.pipelines import GenerativeQAPipeline
from haystack.utils import print_answers

doc_dir = "C:/dataHaystack/tutorial12"
mod_dir = "C:/dataHaystack/Retriev_tutorial12"
BD_dir = "C:/dataHaystack/FAISS-DS_tutorial12"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt12.zip"
# fetch_archive_from_http(url=s3_url, output_dir=doc_dir)
# docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

# SAVED ou pas saved ?
document_store = FAISSDocumentStore.load(BD_dir)
# document_store = FAISSDocumentStore(embedding_dim=128, faiss_index_factory_str="Flat")
# document_store.write_documents(docs)

# SAVED ou pas saved ?
retriever = DensePassageRetriever(mod_dir)
# retriever = DensePassageRetriever(
  #  document_store=document_store,
  #  query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki",
  #  passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
# )
# document_store.update_embeddings(retriever)
# retriever.save(mod_dir)

# Sanity
p_retrieval = DocumentSearchPipeline(retriever)
res = p_retrieval.run(query="Tell me something about Arya Stark?", params={"Retriever": {"top_k": 10}})
print_documents(res, max_text_len=512)

generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa")
generator.save()
#******************************PIPE

pipe = GenerativeQAPipeline(generator, retriever)
#******************************LINE

prediction1 = pipe.run(query="How did Arya Stark's character get portrayed in a television adaptation?", params={"Retriever": {"top_k": 3}})
prediction2 = pipe.run(query="Why is Arya Stark an unusual character?", params={"Retriever": {"top_k": 3}})

print_answers(prediction1, details="minimum")
print_answers(prediction2, details="medium")