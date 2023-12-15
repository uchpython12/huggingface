tokenized_text = "Jim Henson was a puppeteer".split()
print(tokenized_text)


from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


print(tokenizer("Using a Transformer network is simple"))
# 保存標記器(tokenizer)與保存模型相同:
tokenizer.save_pretrained("directory_on_my_computer")