from evaluate import evaluateUsers
from loadData import prepareData, get_max_len
import pandas as pd
import random
import torch
from model import EncoderRNN, AttnDecoderRNN


SOS_token = 0
EOS_token = 1

data_path = 'data\Data_1.xlsx'
input_lang, output_lang, pairs,non_norm_pairs = prepareData('word', 'text', False, data_path)
input_max_len, output_max_len = get_max_len(non_norm_pairs)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1,max_length=output_max_len).to(device)

encoder1.load_state_dict(torch.load(r'C://Users//ir749//Desktop//encoder1.pth'))
decoder1.load_state_dict(torch.load(r'C://Users//ir749//Desktop//decoder2.pth'))

# terminal 

def evaluate(encoder, decoder,input_lang,output_lang,sentence, max_length=int):
    with torch.no_grad():
        # input_tensor = tensorFromSentence_split(input_lang, sentence)
        input_tensor = torch.tensor([input_lang.word2index[word] for word in sentence], dtype=torch.long, device=device).view(-1, 1)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
              
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateUsers(encoder, decoder,input_lang,output_lang,max_length = int):
  words= ''
  words =input('input word: ').split(' ')
  print("Input Word is : ",words)
  output_words, attentions = evaluate(encoder, decoder,input_lang,output_lang,words,max_length=max_length)
  print(output_words)
  output_sentence = ' '.join(output_words)
  output = output_lang.sent2word[output_sentence]
  print("Predict Sentence is :\n ", output)
  return output


result = evaluateUsers(encoder1,decoder1,input_lang,output_lang, max_length = output_max_len)
print(result)