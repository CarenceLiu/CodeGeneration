'''
2024-5-14
lwr
copy from https://github.com/microsoft/CodeXGLUE/blob/main/Text-Code/text-to-code/code/dataset.py
used to load dataset
'''

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import os.path
import pickle
import json
import config

import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence


from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)


class concodeDataset(Dataset):
    def __init__(self, tokenizer, data_dir_path, file_type='train'):
        self.max_len = config.max_len
        self.mode = file_type

        file_path = os.path.join(data_dir_path, file_type+".json")
        cache_file_path = os.path.join(data_dir_path, file_type+"_cache.dat")

        if self.mode != 'test' and os.path.exists(cache_file_path):
            with open(cache_file_path, 'rb') as handle:
                data = pickle.load(handle)
                self.input_nl = data["input_nl"]
                self.output_code = data["output_code"]

        else:
            self.input_nl = []
            self.output_code = []

            datas = open(file_path).readlines()

            length = len(datas)
            for idx, x in enumerate(datas):
                x = json.loads(x)
                code = [tokenizer.bos_token_id]+tokenizer.encode(x["code"][:self.max_len-2])+[tokenizer.eos_token_id]
                nl = [tokenizer.bos_token_id]+tokenizer.encode(x["nl"][:self.max_len-2])+[tokenizer.eos_token_id]
                # if len(code) < self.max_len:
                #     code = code+[tokenizer.pad_token_id for _ in range(self.max_len-len(code))]
                # if len(nl) < self.max_len:
                #     nl = nl+[tokenizer.pad_token_id for _ in range(self.max_len-len(nl))]

                self.input_nl.append(nl)
                self.output_code.append(code)

            if self.mode != 'test':
                with open(cache_file_path, 'wb') as handle:
                    pickle.dump({'input_nl': self.input_nl, 'output_code': self.output_code}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.input_nl)

    def __getitem__(self, item):
        return torch.tensor(self.input_nl[item]), torch.tensor(self.output_code[item])

def collate_fn(data):
    inputs, outputs = zip(*data)
    max_input_len = max(len(seq) for seq in inputs)
    max_output_len = max(len(seq) for seq in outputs)
    pad_input_len = ((max_input_len+7)//8)*8
    pad_output_len = ((max_output_len+7)//8)*8
    inputs = [torch.nn.functional.pad(seq, (0, pad_input_len - len(seq))) for seq in inputs]
    outputs = [torch.nn.functional.pad(seq, (0, pad_output_len - len(seq))) for seq in outputs]
    inputs = pad_sequence(inputs, batch_first=True)
    outputs = pad_sequence(outputs, batch_first=True)
    return inputs, outputs

if __name__ == "__main__":
    # real vocab length = 30526
    tokenizer = BertTokenizer.from_pretrained('../config', add_special_tokens=True)
    # tokenizer.add_special_tokens({
    #     "bos_token": "<s>",
    #     "eos_token": "</s>",
    #     "sep_token": "concode_elem_sep"
    # })
    # tokenizer.add_tokens(["concode_field_sep"])
    # tokenizer.save_pretrained('../config/')
    # tokenizer = BertTokenizer.from_pretrained('../config')
    # print(len(tokenizer.vocab))
    print(tokenizer.bos_token_id)
    print(tokenizer.eos_token_id)
    print(tokenizer.pad_token_id)
    print(tokenizer.unk_token_id)
    print(tokenizer.sep_token_id)
    print(tokenizer.cls_token_id)
    # print(tokenizer.vocab["concode_field_sep"])
    dataset = concodeDataset(tokenizer, "../dataset", file_type='train')
    dataloader = DataLoader(dataset, shuffle=True, batch_size=config.batch_size, drop_last=True, collate_fn=collate_fn)

    for idx, (input_, output_) in enumerate(dataloader):
        print(input_.size()[1])
        print(output_.size()[1])
        break