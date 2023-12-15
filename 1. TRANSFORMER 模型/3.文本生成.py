from transformers import pipeline

generator = pipeline("text-generation")
output = generator("In this course, we will teach you how to")

print(output)

# 您可以使用參數 num_return_sequences 控制生成多少個不同的序列，並使用參數 max_length 控制輸出文本的總長度。