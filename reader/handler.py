import os 
import json
import io
import copy
import random
from functools import wraps
import yaml
from config import inject_config
from utils import measure, jsonify
from reader import Reader






class ModelGenerator(object):
    """
    This class takes as input a question and return its response from the provided
    articles
    """
    
    @measure
    def __init__(self):
        print("os pwd : ",os. getcwd())
        print("os ls : ",os. listdir())
        
        
        self.reader = Reader()
        self.reader.load()
        self.mapping = None
        self.device = None
        self.initialized = False
        print("inference ready!!!!!!!!!!!!")

    @measure
    def initialize(self, ctx):
        """First try to load torchscript else load eager mode state_dict based model"""
        
        properties = ctx.system_properties
        self.initialized = True

    
    

    @measure
    def preprocess(self, data):
        """
         Preprocess received data
        """
        
        self.question = bytes(data[0]["question"]).decode()
        self.context = bytes(data[0]["context"]).decode()
        #print("question : ",self.question)
        return {"question" : self.question , "context" : self.context}

    @measure
    def inference(self, data):
        ''' 
        inference
        '''
        result = self.reader.pipeline(data["question"],data["context"])     
        return result

    


    @measure
    def postprocess(self, inference_output):
        """
        Postprocess
        """
        return jsonify(inference_output)
        

_service = ModelGenerator()

@measure
def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data
