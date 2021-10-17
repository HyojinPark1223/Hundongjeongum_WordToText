
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import numpy as np
import pandas as pd

from konlpy.tag import Hannanum

han = Hannanum()
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.sent2word = {}
        self.n_words = 2

    def addOutputSentence(self, sentence):
        for word in han.morphs(sentence):
            self.addWord(word)

    def addInputword(self, sentence):
        for word in sentence.split(', '):
            self.addWord(word)

    def makedict(self,sentence):
        string = ' '.join(han.morphs(sentence))
        self.sent2word[string] = sentence
        

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
'''
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s
'''
#/content/Data.xlsx
def get_max_len(pairs):
  data = np.array(pairs)
  word_data = data[:,1]
  text_data = data[:,0]

  text = []
  word = []

  for i in range(len(data)):
    text.append(han.morphs(text_data[i]))
    word.append(word_data[i].split(', '))
  text_max_len = max(len(item) for item in text)
  word_max_len = max(len(item) for item in word)
  print(f'text_max_len = {text_max_len}\nword_max_len = {word_max_len}')
  return word_max_len, text_max_len

def readLangs(lang1, lang2, reverse=False, data = list):
    print("Reading lines...") 
    data = data.values

    Non_norm_pairs = [[s for s in l] for l in data]
    # ��� ���� ������ �и��ϰ� ����ȭ
    #pairs = [[normalizeString(s) for s in l] for l in data]
    pairs = Non_norm_pairs
    # ���� ������, Lang �ν��Ͻ� ����
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs, Non_norm_pairs

# def filterPair(p):
#     return len(p[0].split(' ')) < MAX_LENGTH and \
#         len(p[1].split(' ')) < MAX_LENGTH and \
#         p[1].startswith(eng_prefixes)


# def filterPairs(pairs):
#     return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False,path = str):
    data = pd.read_excel(path)
    data = data[data.columns[:2]][:-1]
    input_lang, output_lang, pairs, non_pairs = readLangs(lang1, lang2, reverse, data)
    print("++++++++++readLANGS++++++++",pairs[:3])
    print("Read %s sentence pairs" % len(pairs))
    #pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addInputword(pair[1])
        output_lang.makedict(pair[0])
        output_lang.addOutputSentence(pair[0])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    #print("find_max_pairlen(pairs)",find_max_pairlen(pairs))
    #print(pairs)
    return input_lang, output_lang, pairs, non_pairs