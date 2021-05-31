from transformers import AutoTokenizer, AutoModelForQuestionAnswering , DPRContextEncoder
tokenizer = AutoTokenizer.from_pretrained("ahmedattia143/roberta_squadv1_base")
model = AutoModelForQuestionAnswering.from_pretrained("ahmedattia143/roberta_squadv1_base")
model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
model = DPRContextEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")