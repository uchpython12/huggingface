from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
out =ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")

print(out)

# 在這裡，模型正確地識別出 Sylvain 是一個人 (PER)，Hugging Face 是一個組織 (ORG)，而布魯克林是一個位置 (LOC)。