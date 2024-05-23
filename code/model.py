'''
2024-5-14
lwr
Basic Transformer for code generation

'''
import torch
from torch import nn
import config

class PositionalEncoding(nn.Module):
    '''
    Positional Encoding Module in Transformer
    '''
    def __init__(self, device):
        '''
        pre calculate
        '''
        super().__init__()
        # encoding: [max_len(256), model_dim]
        self.encoding = torch.zeros(config.max_len, config.model_dim, device=device, requires_grad=False)

        pos = torch.arange(0, config.max_len, device=device).float()
        pos = pos.unsqueeze(dim=1)

        _2i = torch.arange(0, config.model_dim, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos/(10000)**(_2i/config.model_dim))
        self.encoding[:, 1::2] = torch.cos(pos/(10000)**(_2i/config.model_dim))
    
    def forward(self, x):
        # input: [batch size, seq_len]
        _, seq_len = x.size()
        
        return self.encoding[:seq_len, :]
    

class MultiHeadAttention(nn.Module):
    '''
    MultiHead Attention(used in Both Decoder and Encoder)
    '''
    def __init__(self):
        super().__init__()
        self.model_dim = config.model_dim
        self.head_num = config.head_num
        self.split_dim = self.model_dim//self.head_num
        self.max_len = config.max_len

        self.softmax = nn.Softmax(dim=-1)
        self.w_q = nn.Linear(self.model_dim, self.model_dim)
        self.w_k = nn.Linear(self.model_dim, self.model_dim)
        self.w_v = nn.Linear(self.model_dim, self.model_dim)
        self.w_concat = nn.Linear(self.model_dim, self.model_dim)
    
    def split(self, x):
        # input: [batch_size, seq_len, model_dim]
        bs, seq_len, _ = x.size()
        x = x.view(bs, seq_len, self.head_num, self.split_dim).transpose(1,2)
        
        #output: [batch_size, head_num, max_len, split_dim]
        return x
    
    def forward(self, q, k, v, mask=None):
        bs, seq_len, _ = v.size()

        # get q, k, v
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # split for multi head
        q, k, v = self.split(q), self.split(k), self.split(v)

        # q, k, v: [batch_size, head_num, max_len, split_dim]
        # attention
        attention = (q @ k.transpose(2,3)) / 8

        # attention: [batch_size, head_num, max_len, max_len]

        if mask != None:
            attention = attention.masked_fill(mask == False, -99999)
        
        attention = self.softmax(attention)
        v = attention @ v

        # attention: [batch_size, head_num, max_len, max_len]
        # v: [batch_size, head_num, max_len, split_dim]

        #concat
        _, _, length, _ = v.size()
        v = v.transpose(1,2).contiguous().view(bs, length, self.model_dim)
        
        # v: [batch_size, max_len, model_dim]
        v = self.w_concat(v)

        return v

class PositionwiseFeedForward(nn.Module):
    '''
    MLP
    '''
    def __init__(self):
        super().__init__()
        self.model_dim = config.model_dim
        self.ffn_hidden = config.ffn_hidden
        self.linear1 = nn.Linear(self.model_dim, self.ffn_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=config.drop_prob)
        self.linear2 = nn.Linear(self.ffn_hidden, self.model_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = config.eps
    
    def forward(self, x):
        #input: [batch_size, seq_len, model_dim]
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean)/torch.sqrt(var+self.eps)
        return x


class TokenEmbedding(nn.Embedding):
    def __init__(self):
        super().__init__(config.voc_size, config.model_dim, padding_idx=config.pad_token_id)

class Embedding(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.token_embedding = TokenEmbedding()
        self.position_embedding = PositionalEncoding(device)
        self.dropout = nn.Dropout(p=config.drop_prob)
    
    def forward(self, x):
        embedding = self.token_embedding(x)
        position = self.position_embedding(x)

        x = self.dropout(embedding+position)
        return x


class EncoderLayer(nn.Module):
    '''
    Encoder Layer
    '''
    def __init__(self):
        super().__init__()
        self.attention = MultiHeadAttention()
        self.norm1 = LayerNorm()
        self.dropout1 = nn.Dropout(p=config.drop_prob)

        self.ffn = PositionwiseFeedForward()
        self.norm2 = LayerNorm()
        self.dropout2 = nn.Dropout(p=config.drop_prob)
    
    def forward(self, x, mask):
        x_ = x
        # attention
        x = self.attention(x, x, x, mask)

        # add&norm
        x = self.dropout1(x)
        x = self.norm1(x+x_)

        x_ = x
        # ffn
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x+x_)

        return x

class Encoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.layer_num = config.layer_num
        self.embedding = Embedding(device) 
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(self.layer_num)])

    def forward(self, x, mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


class DecoderLayer(nn.Module):
    '''
    Decoder Layer
    '''
    def __init__(self):
        super().__init__()
        self.de_attention = MultiHeadAttention()
        self.norm1 = LayerNorm()
        self.dropout1 = nn.Dropout(p=config.drop_prob)

        self.en_de_attention = MultiHeadAttention()
        self.norm2 = LayerNorm()
        self.dropout2 = nn.Dropout(p=config.drop_prob)

        self.ffn = PositionwiseFeedForward()
        self.norm3 = LayerNorm()
        self.dropout3 = nn.Dropout(p=config.drop_prob)
    
    def forward(self, enc_v, dec_v, enc_mask, dec_mask):
        # decoder self attention
        dec_v_ = dec_v

        dec_v = self.de_attention(dec_v, dec_v, dec_v, dec_mask)
        dec_v = self.dropout1(dec_v)
        dec_v = self.norm1(dec_v_+dec_v)

        # encoder decoder attention
        if enc_v != None:
            dec_v_ = dec_v
            dec_v = self.en_de_attention(dec_v, enc_v, enc_v, enc_mask)
            dec_v = self.dropout2(dec_v)
            dec_v = self.norm2(dec_v_+dec_v)
        
        # ffn
        dec_v_ = dec_v
        dec_v = self.ffn(dec_v)
        dec_v = self.dropout3(dec_v)
        dec_v = self.norm3(dec_v_+dec_v)

        return dec_v

class Decoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.layer_num = config.layer_num
        self.target_voc_size = config.voc_size
        self.embedding = Embedding(device) 
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(self.layer_num)])
        self.linear = nn.Linear(config.model_dim, self.target_voc_size)

    def forward(self, enc_v, dec_v,  enc_mask, dec_mask):

        dec_v = self.embedding(dec_v)

        for layer in self.layers:
            dec_v = layer(enc_v, dec_v, enc_mask, dec_mask)
        
        dec_v = self.linear(dec_v)
        
        return dec_v

class Transformer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id
        self.unk_token_id = config.unk_token_id
        self.sep_token_id = config.sep_token_id
        self.device = device
        self.encoder = Encoder(device)
        self.decoder = Decoder(device)

    def forward(self, input_, output_):
        # print("input size:", input_.size())
        # print("output size:", output_.size())
        input_mask = self.src_mask(input_)
        output_mask = self.trg_mask(output_)
        # print("input mask size:", input_mask.size())
        # print("output mask size:", output_mask.size())
        enc_out = self.encoder(input_, input_mask)
        output_ = self.decoder(enc_out, output_, input_mask, output_mask)

        return output_
    
    def src_mask(self, x):
        mask = (x != self.pad_token_id).unsqueeze(1).unsqueeze(2).bool()
        return mask

    def trg_mask(self, x):
        mask = (x != self.pad_token_id).unsqueeze(1).unsqueeze(3).bool()
        seq_len = x.size()[1]
        sub_mask = torch.tril(torch.ones(seq_len, seq_len)).type(torch.ByteTensor).to(self.device).bool()
        mask = mask&sub_mask
        return mask