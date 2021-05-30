from functools import wraps
import yaml
from abc import ABC,ABCMeta, abstractmethod
from yacs.config import CfgNode as CN

_C = CN()

_C.reader = CN()
#_C.reader.model_path = "deepset/roberta-base-squad2"
_C.reader.model_path = "ahmedattia143/roberta_squadv1_base"

_C.reader.top_k = 1


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

