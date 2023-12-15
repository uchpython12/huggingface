from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

# 輸出本身是一個包含兩個鍵的字典，input_ids和attention_mask。
# input_ids包含兩行整數（每個句子一行），它們是每個句子中標記的唯一標記（token）。
# 我們將在本章後面解釋什麼是attention_mask。

# 瀏覽模型

from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)

# 高維向量？
# Transformers 模塊的向量輸出通常較大。它通常有三個維度：
#
# Batch size: 一次處理的序列數（在我們的示例中為2）。
# Sequence length: 序列的數值表示的長度（在我們的示例中為16）。
# Hidden size: 每個模型輸入的向量維度。

outputs = model(**inputs)
print(outputs.last_hidden_state.shape)

# 🤗 Transformers中有許多不同的體系結構，每種體系結構都是圍繞處理特定任務而設計的。以下是一個非詳盡的列表：
#
# *Model (retrieve the hidden states)
# *ForCausalLM
# *ForMaskedLM
# *ForMultipleChoice
# *ForQuestionAnswering
# *ForSequenceClassification
# *ForTokenClassification
# 以及其他 🤗

from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)

print(outputs.logits.shape)

print(outputs.logits)

import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

model.config.id2label