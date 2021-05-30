from config import inject_config
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

class Reader():

  @inject_config
  def get_tokenizer(self, config = None):
    self.tokenizer = AutoTokenizer.from_pretrained(config.reader.model_path)
    return self.tokenizer
  
  @inject_config
  def get_model(self, config = None):
    self.model = AutoModelForQuestionAnswering.from_pretrained(config.reader.model_path)
    return self.model

  @inject_config
  def get_pipeline(self, config = None):
    tokenizer = self.get_tokenizer()
    model = self.get_model() 
    self.pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)
    return self.pipeline

  def load(self,):
    self.get_pipeline()

  @inject_config
  def run(self,question,context , config =None):
    return self.pipeline(
        question=question,
        context=context,
        topk=config.reader.top_k)
