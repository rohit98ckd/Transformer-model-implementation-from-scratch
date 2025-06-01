import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter

# Tokenize the sentence and create vocabulary
def tokenize_and_build_vocab(sentence):
    # Split sentence into words (simple tokenization)
    tokens = sentence.lower().split()
    # Create vocabulary (unique words to indices)
    vocab = {word: idx + 1 for idx, word in enumerate(set(tokens))}  # Start indices from 1 (0 for padding)
    vocab['<PAD>'] = 0  # Padding token
    # Convert sentence to indices
    token_ids = [vocab[word] for word in tokens]
    return token_ids, vocab

# Positional Encoding: Adds info about where each word is in a sentence
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# Visualize Positional Encoding with sentence-specific annotations
def visualize_positional_encoding(d_model, seq_len, tokens):
    pe = PositionalEncoding(d_model, seq_len)
    pe_matrix = pe.pe.squeeze(0).numpy()[:seq_len, :]  # Shape: (seq_len, d_model)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pe_matrix, cmap='coolwarm', cbar_kws={'label': 'Encoding Value'})
    plt.title('Positional Encoding for Sentence: "My name is Rohit ..."', fontsize=14)
    plt.xlabel('Embedding Dimension (Features)', fontsize=12)
    plt.ylabel('Word Position', fontsize=12)
    plt.yticks(np.arange(0.5, seq_len, 1), tokens, rotation=0)
    plt.xticks(np.arange(0, d_model, 8), rotation=0)
    plt.text(0.5, -0.5, 'Blue = Negative, Red = Positive\nEncodes word positions in sentence', 
             fontsize=10, color='black', ha='left')
    plt.savefig('positional_encoding_sentence.png', bbox_inches='tight')
    plt.close()

# Scaled Dot-Product Attention: How words pay attention to each other
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output, attn

# Visualize Attention Weights with sentence-specific labels
def visualize_attention(attn, title, tokens, batch_idx=0, head_idx=0):
    attn = attn[batch_idx, head_idx].detach().numpy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn, cmap='viridis', annot=True, fmt='.2f', cbar_kws={'label': 'Attention Score'})
    plt.title(f'{title}\n(How Words in "{tokens[0]} ..." Pay Attention)', fontsize=14)
    plt.xlabel('Words in Sentence (Keys)', fontsize=12)
    plt.ylabel('Words in Sentence (Queries)', fontsize=12)
    plt.xticks(np.arange(0.5, len(tokens), 1), tokens, rotation=45, ha='right')
    plt.yticks(np.arange(0.5, len(tokens), 1), tokens, rotation=0)
    plt.text(0.5, -0.5, 'Yellow = High Attention, Purple = Low Attention', 
             fontsize=10, color='black', ha='left')
    plt.savefig(f'{title.lower().replace(" ", "_")}_sentence.png', bbox_inches='tight')
    plt.close()

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        output, attn = self.attention(Q, K, V, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.W_o(output)
        return output, attn

# Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, attn = self.mha(x, x, x, mask)
        x = self.layernorm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + self.dropout(ffn_output))
        return x, attn

# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn1_output, attn1 = self.mha1(x, x, x, tgt_mask)
        x = self.layernorm1(x + self.dropout(attn1_output))
        attn2_output, attn2 = self.mha2(x, enc_output, enc_output, src_mask)
        x = self.layernorm2(x + self.dropout(attn2_output))
        ffn_output = self.ffn(x)
        x = self.layernorm3(x + self.dropout(ffn_output))
        return x, attn1, attn2

# Transformer Model
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=64, num_heads=4, num_layers=2, d_ff=256, max_len=100, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, tokens=None):
        src_emb = self.dropout(self.pos_encoder(self.encoder_embedding(src)))
        enc_output = src_emb
        enc_attns = []
        for i, layer in enumerate(self.encoder_layers):
            enc_output, attn = layer(enc_output, src_mask)
            enc_attns.append(attn)
            visualize_attention(attn, f'Encoder Layer {i+1} Self-Attention', tokens)
        
        tgt_emb = self.dropout(self.pos_encoder(self.decoder_embedding(tgt)))
        dec_output = tgt_emb
        dec_self_attns, dec_enc_attns = [], []
        for i, layer in enumerate(self.decoder_layers):
            dec_output, self_attn, enc_attn = layer(dec_output, enc_output, src_mask, tgt_mask)
            dec_self_attns.append(self_attn)
            dec_enc_attns.append(enc_attn)
            visualize_attention(self_attn, f'Decoder Layer {i+1} Self-Attention', tokens)
            visualize_attention(enc_attn, f'Decoder Layer {i+1} Encoder-Decoder Attention', tokens)
        
        output = self.fc(dec_output)
        return output, enc_attns, dec_self_attns, dec_enc_attns

# Create masks
def create_padding_mask(seq, pad_idx=0):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def create_lookahead_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return (~mask).unsqueeze(0).unsqueeze(0)

# Visualize Transformer architecture
def visualize_transformer_architecture(tokens):
    plt.figure(figsize=(12, 8))
    plt.text(0.5, 0.95, 'Transformer for "My name is Rohit ..."', fontsize=16, ha='center', weight='bold')
    plt.text(0.25, 0.8, 'Encoder\n(Reads Input)', fontsize=12, ha='center', bbox=dict(facecolor='lightblue', alpha=0.5))
    plt.text(0.75, 0.8, 'Decoder\n(Writes Output)', fontsize=12, ha='center', bbox=dict(facecolor='lightgreen', alpha=0.5))
    plt.arrow(0.25, 0.7, 0, -0.2, head_width=0.02, color='blue')
    plt.arrow(0.75, 0.7, 0, -0.2, head_width=0.02, color='green')
    plt.text(0.25, 0.6, 'Multi-Head\nAttention', fontsize=10, ha='center')
    plt.text(0.25, 0.5, 'Feed-Forward\nNetwork', fontsize=10, ha='center')
    plt.text(0.75, 0.6, 'Masked\nMulti-Head\nAttention', fontsize=10, ha='center')
    plt.text(0.75, 0.5, 'Encoder-Decoder\nAttention', fontsize=10, ha='center')
    plt.text(0.75, 0.4, 'Feed-Forward\nNetwork', fontsize=10, ha='center')
    plt.text(0.5, 0.3, f'Input: {tokens[0]} {tokens[1]} ... → Encoder → Decoder → Output', fontsize=12, ha='center')
    plt.axis('off')
    plt.savefig('transformer_architecture_sentence.png', bbox_inches='tight')
    plt.close()

# Example usage with specific sentence
def main():
    # Input sentence
    sentence = "my name is rohit and i study in iit patna"
    token_ids, vocab = tokenize_and_build_vocab(sentence)
    tokens = sentence.lower().split()
    seq_len = len(tokens)  # 10 words
    src_vocab_size = len(vocab)  # 10 unique words + padding
    tgt_vocab_size = src_vocab_size  # Same vocab for simplicity
    d_model = 64
    num_heads = 4
    num_layers = 2
    d_ff = 256
    batch_size = 1

    # Initialize model
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff)
    
    # Convert sentence to tensor
    src = torch.tensor([token_ids], dtype=torch.long)  # Shape: (1, seq_len)
    tgt = torch.tensor([token_ids], dtype=torch.long)  # Same for simplicity
    
    # Create masks
    src_mask = create_padding_mask(src)
    tgt_mask = create_lookahead_mask(seq_len) & create_padding_mask(tgt)
    
    # Forward pass
    output, enc_attns, dec_self_attns, dec_enc_attns = model(src, tgt, src_mask, tgt_mask, tokens)
    
    # Generate visualizations
    visualize_positional_encoding(d_model, seq_len, tokens)
    visualize_transformer_architecture(tokens)
    
    print("Model output shape:", output.shape)
    print("Vocabulary:", vocab)
    print("Visualizations saved: positional_encoding_sentence.png, transformer_architecture_sentence.png, and attention heatmaps")

if __name__ == "__main__":
    main()