import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import PennTreebank
import math
import os
from ptb_modeling import TransformerModel
#from ptb import TransformerModel

tokenizer = get_tokenizer('basic_english')
train_iter = PennTreebank(split='train')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

ntokens = len(vocab)  # size of vocabulary
emsize = 1024  # embedding dimension
d_hid = 1024  # dimension of the feedforward network model
nlayers = 1  # number of nn.TransformerEncoderLayer
nhead = 1  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
model.load_state_dict(torch.load("result/best_model_params.pt"))
model.eval()

def predict_next_word(text, model, vocab, device):
    tokens = tokenizer(text)
    indices = torch.tensor([vocab[token] for token in tokens], dtype=torch.long).unsqueeze(1).to(device)
    with torch.no_grad():
        output = model(indices)
    last_word_logits = output[-1]
    predicted_index = torch.argmax(last_word_logits, dim=1).item()
    itos = vocab.get_itos() 
    predicted_word = itos[predicted_index]
    return predicted_word

if __name__ == "__main__":
    text = "Turmoil in Beijing but"
    print(text + '.')
    next_word = ''
    for i in range(20):
        next_word = predict_next_word(text, model, vocab, device)
        text = text + ' ' + next_word
        print(text + '.')