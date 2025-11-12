import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size: int = 256 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 4 # number of layers
    n_head: int = 6 # number of heads
    n_embed: int = 384 # embedding dimension

class MultiHeadAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.n_embed = config.n_embed
        self.n_head = config.n_head
        assert self.n_embed % self.n_head == 0
        # it's all 3: Q, K, V in one go for efficiency.
        self.attn = nn.Linear(self.n_embed,3*self.n_embed)
        self.w_o = nn.Linear(self.n_embed , self.n_embed)
        self.w_o.NANOGPT_SCALE_INIT = 1
        self.register_buffer("mask",torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size))


    def forward(self,x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality 
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        qkv = self.attn(x)
        #splits the tensor into equal chunks, each of size split_size along the dimension dim split_size = n_embed
        q,k,v = qkv.split(self.n_embed, dim=2)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (B, nh, T, hs)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (B, nh, T, hs)


        att = (q @ k.transpose(-2,-1)) * (1/math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.w_o(y)
        return y


class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.linear_1 = nn.Linear(config.n_embed , 4*config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.linear_2 = nn.Linear(4*config.n_embed,config.n_embed)
        self.linear_2.NANOGPT_SCALE_INIT = 1

    def forward(self,x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x 


class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x 

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed),
        ))

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # initialization
        self.apply(self._init_weights) #nn.Module.apply(fn) walks the entire model tree (all submodules recursively) and calls your function fn(module) on every layer.

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size , f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        tok_emb = self.transformer.wte(idx)
        pos = torch.arange(0, T, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)

        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),targets.view(-1)) # # Cross entropy is applied for each token position CE_i = - log(softmax(logits[i])[ targets[i] ])
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # crop to block size
            idx_cond = idx[:, -self.config.block_size:]

            # forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx










