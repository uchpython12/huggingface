from transformers import pipeline

unmasker = pipeline("fill-mask")
out = unmasker("This course will teach you all about <mask> models.", top_k=2)

print(out)
