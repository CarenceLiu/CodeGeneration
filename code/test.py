'''
2024-5-14
lwr

'''

import config
from dataset import concodeDataset, collate_fn
from model import Transformer
import math
import time
import collections

import torch
from torch import nn, optim
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from bleu import compute_bleu

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

def batch_tensor_to_seq(x, idx, tokenizer):
    bs, seq_len = x.size()
    seq_list = []
    for i in range(seq_len):
        if x[idx,i] == tokenizer.eos_token_id:
            return tokenizer.decode(seq_list)
        else:
            if x[idx,i] != tokenizer.bos_token_id and x[idx,i] != tokenizer.pad_token_id and x[idx,i] != tokenizer.unk_token_id and x[idx,i] != tokenizer.sep_token_id and x[idx,i] != tokenizer.cls_token_id:
                seq_list.append(x[idx,i])
    return tokenizer.decode(seq_list)



temperature = 0.95
tokenizer = BertTokenizer.from_pretrained('../config', add_special_tokens=True)
test_dataset = concodeDataset(tokenizer, "../dataset", file_type='test')
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=config.batch_size, drop_last=True, collate_fn=collate_fn)

dev = torch.device("cpu")
if torch.cuda.is_available():
    dev = torch.device("cuda")
    print("GPU")
else:
    print("CPU")
model = Transformer(dev)
model.load_state_dict(torch.load('../result/model.pth'))
model = model.to(dev)


if __name__ == "__main__":
    model.eval()
    total_bleu_score = 0.0
    exact_match = 0
    total_size = 0
    for batch_idx, (input_, output_) in enumerate(test_dataloader):
        bs, seq_len = input_.size()
        total_size += bs
        prediction_seq_ = torch.zeros((bs, config.max_len), dtype=torch.int)
        prediction_seq_[:, 0] = tokenizer.bos_token_id
        predict_len = 1
        prediction_mask = torch.zeros((bs, config.max_len), dtype=torch.bool)
        prediction_mask[:, 0] = True
        input_, output_, prediction_seq_ = input_.to(dev), output_.to(dev), prediction_seq_.to(dev)
        prediction_mask = prediction_mask.to(dev)

        # prediction
        for i in range(1, config.max_len):
            prediction_seq_ = prediction_seq_.masked_fill(prediction_mask == False, tokenizer.pad_token_id)
            # print(input_)
            # print(prediction_mask)
            # print(prediction_seq_)
            predict_result = model(input_, prediction_seq_)
            predict_result = predict_result/temperature
            probs = torch.nn.functional.softmax(predict_result, dim=-1)
            # word_indices = torch.argmax(predict_result[:, i, :], dim=1)
            next_word_indices = torch.multinomial(probs[:, i, :], num_samples=1)
            # print(next_word_indices)
            prediction_seq_[:, i] = next_word_indices[:, 0]
            prediction_mask[: i] = True 

        codes = []
        predict_codes = []
        for i in range(bs):
            nl = batch_tensor_to_seq(input_, i, tokenizer)
            code = batch_tensor_to_seq(output_, i, tokenizer)
            predict_code = batch_tensor_to_seq(prediction_seq_, i, tokenizer)
            if i == 0:
                print(nl)
                print(code)
                print(predict_code)
            if code == predict_code:
                print("EM: ", code)
                exact_match += 1
            codes.append(code.strip().split())
            predict_codes.append(predict_code.strip().split())

        bleu_score, _, _, _, _, _ = compute_bleu(codes, predict_code)
        print("batch idx: %d, bleu score: %f" %(batch_idx, bleu_score))
        total_bleu_score += bleu_score
    
    bleu_score = total_bleu_score/total_size
    exact_match_score = exact_match/total_size
    print("bleu score: %f, em score: %s" %(bleu_score*100, exact_match_score*100))
    


        
