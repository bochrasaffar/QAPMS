from functools import wraps
import yaml
from abc import ABC,ABCMeta, abstractmethod
from yacs.config import CfgNode as CN

_C = CN()

_C.dense_retriever = CN()
_C.dense_retriever.ce_path = "facebook/dpr-ctx_encoder-single-nq-base"
_C.dense_retriever.qe_path = "facebook/dpr-question_encoder-single-nq-base"
_C.dense_retriever.name = "DPR"
_C.dense_retriever.max_seq_len_query=64
_C.dense_retriever.max_seq_len_passage=256
_C.dense_retriever.batch_size=16
_C.dense_retriever.top_k = 5

_C.sparse_retriever = CN()
_C.sparse_retriever.name = "BM25"
_C.sparse_retriever.top_k = 5

_C.reader = CN()
#_C.reader.model_path = "deepset/roberta-base-squad2"
_C.reader.model_path = "ahmedattia143/roberta_squadv1_base"

_C.reader.top_k = 5

_C.evaluator = CN()
_C.evaluator.retriever_metric = ["recall" , "map"]
_C.evaluator.reader_metric = ["f1" , "precision" , "recall"]

_C.preprocessor = None
_C.seed = 42
_C.device = "cpu"

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    #return _C.clone()
    return _C

def dump_cfg(config = get_cfg_defaults() , path = "experiment.yaml"):
    """Save a yacs CfgNode object in a yaml file in path."""
    stream = open(path, 'w')
    stream.write(config.dump())
    stream.close()

def inject_config(funct):
    """Inject a yacs CfgNode object in a function as first arg."""
    @wraps(funct)
    def function_wrapper(*args,**kwargs):
        return funct(*args,**kwargs,config=_C)  
    return function_wrapper

def dump_dict(config,path="config.yaml"):
        stream = open(path, 'w')
        yaml.dump(config,stream)
        stream.close()

c=get_cfg_defaults()

