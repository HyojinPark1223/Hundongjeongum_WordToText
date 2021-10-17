import matplotlib.pyplot as plt
import random
import numpy as np
import torch
from torch import optim
from loadData import *
from model import * 
from konlpy.tag import Hannanum 

han = Hannanum()
SOS_token = 0
EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher_forcing_ratio = 0.5

def indexesFromSentence_split(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(', ')]

def indexesFromSentence_token(lang, sentence):
    return [lang.word2index[word] for word in han.morphs(sentence)]



def tensorFromSentence_split(lang, sentence):
    indexes = indexesFromSentence_split(lang, sentence)
    
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorFromSentence_token(lang, sentence):
    indexes = indexesFromSentence_token(lang, sentence)
    indexes.append(EOS_token) 
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence_split(input_lang, pair[1])
    target_tensor = tensorFromSentence_token(output_lang, pair[0])
    return (input_tensor, target_tensor)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=int):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing ����: ��ǥ�� ���� �Է����� ����
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Teacher forcing ������: �ڽ��� ������ ���� �Է����� ���
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # �Է����� ����� �κ��� �����丮���� �и�

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=1000, learning_rate=0.01,max_length = int):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # print_every ���� �ʱ�ȭ
    plot_loss_total = 0  # plot_every ���� �ʱ�ȭ

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion,max_length= max_length)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter%plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            print(plot_losses)
            plt.plot(plot_losses,label = f'Loss : {plot_losses[-1]}')
            plt.title('Training_Loss')
            plt.legend()
            plt.show()


data_path = 'data\Data_2.xlsx'
input_lang, output_lang, pairs,non_norm_pairs = prepareData('word', 'text', False, data_path)
input_max_len, output_max_len = get_max_len(non_norm_pairs)

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1,max_length=output_max_len).to(device)

#trainIters(encoder1, attn_decoder1, 75000, print_every=5000)
losses = trainIters(encoder1, decoder1,input_lang,output_lang,pairs, n_iters = 5000, print_every=1000,plot_every =1000,max_length = output_max_len)

torch.save(encoder1.state_dict(),'encoder2.pth')
torch.save(decoder1.state_dict(),'decoder2.pth')

print('training Done')