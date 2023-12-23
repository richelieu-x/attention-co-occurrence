import torch
import torchtext
from torchtext.datasets import PennTreebank

dataset = PennTreebank(split='test')

def load_penn_treebank_sentences(dataset):
    sentences = []
    for line in dataset:
        sentences.append(line)
    return sentences

def count_cooccurrence_frequency(word1, word2, sentences):
    cooccurrence_count = 0

    for sentence in sentences:
        if word1 in sentence and word2 in sentence:
            cooccurrence_count += 1

    return cooccurrence_count

sentences = load_penn_treebank_sentences(dataset)
#print(sentences[:2])

# 示例用法
word1 = 'new'
word2 = 'york'
cooccurrence_frequency = count_cooccurrence_frequency(word1, word2, sentences)
cooccurrence_score = cooccurrence_frequency / 100#
print(word1, '&', word2)
print(cooccurrence_frequency)
print(cooccurrence_score)