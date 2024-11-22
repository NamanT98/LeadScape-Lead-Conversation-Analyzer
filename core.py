import torch
import torch.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(scores, dim=-1)
        return torch.matmul(attention, value), attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.query(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        key = self.key(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = self.value(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        attn_output, _ = ScaledDotProductAttention()(query, key, value, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.fc(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

class CustomTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_len, num_classes, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, dropout_rate) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids):
        x = self.embedding(input_ids) + self.positional_encoding[:, :input_ids.size(1), :]
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        return self.sigmoid(self.fc(x))
    

class Tokenizer:
    def __init__(self):
        self.word2idx = {"[PAD]": 0, "[UNK]": 1}
        self.idx2word = {0: "[PAD]", 1: "[UNK]"}
        self.word_count = {}

    def fit(self, texts):
        for text_list in texts:
            for text in text_list:  
                for word in text.split():
                    if word not in self.word2idx:
                        idx = len(self.word2idx)
                        self.word2idx[word] = idx
                        self.idx2word[idx] = word
                    self.word_count[word] = self.word_count.get(word, 0) + 1

    def encode(self, text, max_len):
        tokens = [self.word2idx.get(word, 1) for word in text.split()]
        tokens = tokens[:max_len]
        return tokens + [0] * (max_len - len(tokens))

    def vocab_size(self):
        return len(self.word2idx)