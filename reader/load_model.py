from transformers import AutoTokenizer, AutoModelForQuestionAnswering
tokenizer = AutoTokenizer.from_pretrained("ahmedattia143/roberta_squadv1_base")
model = AutoModelForQuestionAnswering.from_pretrained("ahmedattia143/roberta_squadv1_base")