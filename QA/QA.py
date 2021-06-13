from haystack import Finder , Pipeline
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers
from haystack.document_store.faiss import FAISSDocumentStore
from haystack.document_store.memory import InMemoryDocumentStore
from haystack.document_store import ElasticsearchDocumentStore
from haystack.retriever.sparse import TfidfRetriever , ElasticsearchRetriever
from haystack.retriever.dense import DensePassageRetriever
from haystack.preprocessor import PreProcessor
from haystack.eval import EvalReader, EvalRetriever
from haystack.pipeline import ExtractiveQAPipeline
from haystack.pipeline import JoinDocuments

from config import inject_config

class QA:
  def __init__(self):
    None




  @inject_config
  def load_dense_retriever_store(self,path="faiss.index",config=None):
     self.dense_document_store = FAISSDocumentStore(sql_url = "sqlite:///faiss.db",faiss_index_factory_str="Flat")
     self.dense_document_store = self.dense_document_store.load(sql_url="sqlite:///faiss.db",faiss_file_path=path,index="document")
     return self.dense_document_store
  
  @inject_config
  def load_sparse_retriever_store(self,config=None):
     #self.sparse_document_store = ElasticsearchDocumentStore(host="es-7a4a10.es.eastus2.azure.elastic-cloud.com",port="9243", username="elastic", password="hhCSjv0QUUvlWiNFYfja4vO9", index="document",
     #                                       scheme='https',create_index=False, )
     self.sparse_document_store = ElasticsearchDocumentStore(host="pfa-qa.es.us-west1.gcp.cloud.es.io",port="9243", username="elastic", password="VYFT5cdDE6lvqKWBzRiHnZ4k", index="document",
                                            scheme='https',create_index=False, )
     return self.sparse_document_store


  
  @inject_config
  def load_dense_retriever(self,config=None):
    cdr =config.dense_retriever 
    if (cdr is None):
      return None
    else :
      if (cdr.name=="DPR"):

        retriever = DensePassageRetriever(document_store=self.load_dense_retriever_store(),
                                  query_embedding_model=cdr.qe_path,
                                  passage_embedding_model=cdr.ce_path,
                                  max_seq_len_query=cdr.max_seq_len_query,
                                  max_seq_len_passage=cdr.max_seq_len_passage,
                                  batch_size=cdr.batch_size,
                                  use_gpu=config.device=="gpu",
                                  embed_title=True,
                                  use_fast_tokenizers=True)
        #self.dense_document_store.update_embeddings(retriever)
        return retriever
    return None

  @inject_config
  def load_sparse_retriever(self,config=None):
    csr =config.sparse_retriever 
    if (csr is None):
      return None
    else :
      if (csr.name=="BM25"):
        return ElasticsearchRetriever(document_store=self.load_sparse_retriever_store())
    return None
  @inject_config
  def load_reader(self , config = None):
    cr = config.reader
    self.reader = FARMReader(model_name_or_path=cr.model_path, use_gpu=config.device=="gpu")
    return self.reader

  def prepare(self , ):
    self.dense_retriever = self.load_dense_retriever()
    self.sparse_retriever = self.load_sparse_retriever()
    self.reader = self.load_reader()

  

class QAInference(QA):
  @inject_config
  def create_inference_pipeline(self, config = None):
    p = Pipeline()
    p.add_node(component=self.sparse_retriever, name="SparseRetriever", inputs=["Query"])
    p.add_node(component=self.dense_retriever, name="DenseRetriever", inputs=["Query"])
    p.add_node(component=JoinDocuments(join_mode="concatenate"), name="JoinResults", inputs=["SparseRetriever", "DenseRetriever"])
    
    p.add_node(component=self.reader, name="QAReader", inputs=["JoinResults"])
    #p.add_node(component=self.reader, name="QAReader", inputs=["DenseRetriever"])
    #p.add_node(component=self.reader, name="QAReader", inputs=["SparseRetriever"])
    self.inference_pipeline = p
    return p

  def prepare(self):
    super().prepare()
    self.create_inference_pipeline()

  def run(self,question):
    return self.inference_pipeline.run(
        query=question,
        top_k_retriever=8,
        top_k_reader=4)
