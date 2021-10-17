import torch

def evaluate(encoder, decoder,input_lang,output_lang,sentence = list, max_length=int):
    with torch.no_grad():
        # input_tensor = tensorFromSentence_split(input_lang, sentence)
        input_tensor = torch.tensor([input_lang.word2index[word] for word in sentence], dtype=torch.long).view(-1, 1)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size)#), device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[1]])#, device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == 1:
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateUsers(encoder, decoder,input_lang,output_lang, max_length = int):
  words= ''
  words =input('Input word: ').split(' ')
  print("Input word is: ",words)
  output_words, attentions = evaluate(encoder, decoder,input_lang,output_lang,sentence = words,max_length=max_length)
  output_sentence = ' '.join(output_words)
  output = output_lang.sent2word[output_sentence]
  print("Predict Sentence: ", output)
  return output