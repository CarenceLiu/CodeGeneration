'''
2024-5-14
lwr
code for training
'''
import config
from dataset import concodeDataset, collate_fn
from model import Transformer
import math
import time

import torch
from torch import nn, optim
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from bleu import idx_to_word, get_bleu

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)

tokenizer = BertTokenizer.from_pretrained('../config', add_special_tokens=True)
train_dataset = concodeDataset(tokenizer, "../dataset", file_type='train')
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size, drop_last=True, collate_fn=collate_fn)
validation_dataset = concodeDataset(tokenizer, "../dataset", file_type='dev')
validation_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size, drop_last=True, collate_fn=collate_fn)

dev = torch.device("cpu")
if torch.cuda.is_available():
    dev = torch.device("cuda")
    print("GPU")
else:
    print("CPU")
model = Transformer(dev)
model = model.to(dev)
model.apply(initialize_weights)
criterion = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weigh_decay)
scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

if __name__ == "__main__":
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
    train_losses, val_losses = [], []
    log_file = open("../result/train_log.txt","w")
    for i in range(config.epoch):
        model.train()
        total_train_loss = 0
        print("train data batch size: ", len(train_dataloader))
        for batch_idx, (input_, output_) in enumerate(train_dataloader):
            input_, output_ = input_.to(dev), output_.to(dev)
            optimizer.zero_grad()
            prediction_ = model(input_, output_[:, :-1])
            prediction_ = prediction_.contiguous().view(-1, prediction_.shape[-1])

            output_ = output_[:, 1:].contiguous().view(-1)
            loss = criterion(prediction_, output_)
            loss.backward() 
            optimizer.step()

            total_train_loss += loss.item()
            print("[epoch: %d, batch %d], train loss: %f" % (i, batch_idx, loss.item()))
            train_losses.append(loss.item())
        
        print("epoch: %d, train loss: %f" % (i, total_train_loss/len(train_dataloader)))
        log_file.write("epoch: %d, train loss: %f\n" % (i, total_train_loss/len(train_dataloader)))

        # validation 
        model.eval()
        total_validation_loss = 0
        for batch_idx, (input_, output_) in enumerate(validation_dataloader):
            input_, output_ = input_.to(dev), output_.to(dev)
            prediction_ = model(input_, output_[:, :-1])
            prediction_ = prediction_.contiguous().view(-1, prediction_.shape[-1])

            output_ = output_[:, 1:].contiguous().view(-1)
            loss = criterion(prediction_, output_)

            total_validation_loss += loss.item()

        print("epoch: %d, validation loss: %f" % (i, total_validation_loss/len(validation_dataloader)))
        val_losses.append(total_validation_loss/len(validation_dataloader))
        scheduler.step(total_validation_loss/len(validation_dataloader))

        if i >= 2:
            torch.save(model.state_dict(), "../result/model.pth")
            print(train_losses)
            print(val_losses)
            with open('../result/train_loss.txt', 'w') as file:
                for number in train_losses:
                    file.write(str(number) + '\n')
            with open('../result/validation.txt', 'w') as file:
                for number in val_losses:
                    file.write(str(number) + '\n')
    
    # torch.save(model.state_dict(), "../result/model.pth")

    with open('../result/train_loss.txt', 'w') as file:
        for number in train_losses:
            file.write(str(number) + '\n')
    with open('../result/validation.txt', 'w') as file:
        for number in val_losses:
            file.write(str(number) + '\n')
    log_file.close()
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="train loss")
    # plt.plot(val_losses, label="validation loss")
    print(val_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
            