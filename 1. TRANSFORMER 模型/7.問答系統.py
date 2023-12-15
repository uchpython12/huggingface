from transformers import pipeline

question_answerer = pipeline("question-answering")
out = question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)
print(out)
