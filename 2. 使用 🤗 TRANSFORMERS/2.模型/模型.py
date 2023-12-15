from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")
# 保存模型
# model.save_pretrained("directory_on_my_computer")


sequences = ["Hello!", "Cool.", "Nice!"]

encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]

import torch

model_inputs = torch.tensor(encoded_sequences)
output = model(model_inputs)

print(output)