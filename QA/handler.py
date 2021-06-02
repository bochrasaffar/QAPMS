import os 
import json
import io
import copy
import random
import yaml
from config import inject_config
from utils import measure , jsonify , load_json
from QA import QA, QAInference








class ModelGenerator(object):
    """
    This class takes as input a question and return its response from the provided
    articles
    """
    
    @measure
    def __init__(self):
        print("os pwd : ",os. getcwd())
        print("os ls : ",os. listdir())
        self.pipeline = QAInference()
        self.pipeline.prepare()
        self.mapping = None
        self.device = None
        self.initialized = False
        print("inference ready!!!!!!!!!!!!")
        
    

    @measure
    def initialize(self, ctx):
        """First try to load torchscript else load eager mode state_dict based model"""
        
        properties = ctx.system_properties        
        
        #model_dir = properties.get("model_dir")
        #weight_path = os.path.join(model_dir, "best_model.pth")
        self.initialized = True

    

    @measure
    def preprocess(self, data):
        """
         Preprocess image
        """
        if "body" in data[0]:
            data[0] = load_json(bytes(data[0]["body"]).decode())
            self.question = data[0]["question"]
        else:
            self.question = bytes(data[0]["question"]).decode()
        return self.question

    @measure
    def inference(self, question):
        ''' 
        inference
        '''
        result = self.pipeline.run(question)
        return result

    


    @measure
    def postprocess(self, inference_output):
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
