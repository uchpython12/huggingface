from transformers import pipeline


en_zh = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")

print(en_zh("English becomes Chinese"))

zh_en = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")

print(zh_en("中文變成英文"))

# 與文本生成和摘要一樣，您可以指定結果的 max_length 或 min_length。