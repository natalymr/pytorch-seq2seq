import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from spacy.lang.en import English
from spacy.lang.de import German
from tqdm import tqdm
import random
from collections import Counter

from seq2seq.dataset import SourceField, TargetField

# if torch.cuda.is_available:
#     device = torch.device("cuda")
# else:

device = torch.device("cpu")


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, drop_prob=0):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm_del = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=drop_prob, batch_first=True)
        self.lstm_add = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=drop_prob, batch_first=True)

    def forward(self, inputs, hidden, is_del: bool):
        # Embed input words
        embedded = self.embedding(inputs)
        # print(f'encoder: inputs shape = {inputs.size()}')
        # Pass the embedded word vectors into LSTM and return all outputs
        if is_del:
            return self.lstm_del(embedded, hidden)
        else:
            return self.lstm_add(embedded, hidden)
        # output, hidden = self.lstm(embedded, hidden)
        # return output, hidden

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device))


class BahdanauDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, drop_prob=0.1):
        super(BahdanauDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.drop_prob = drop_prob

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        self.fc_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_encoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size))
        self.V = nn.Linear(self.hidden_size, 1)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.drop_prob)
        self.lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size, batch_first=True)
        self.classifier = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs, hidden, encoder_outputs):
        encoder_outputs = encoder_outputs.squeeze()
        # print(f'D-forward = {inputs.size()},  h = {hidden[0].size()}, c={hidden[1].size()},'
        #       f'encoder output = {encoder_outputs.size()}')
        # Embed input words
        embedded = self.embedding(inputs[:, None])
        embedded = self.dropout(embedded)
        # print(f'embedded = {embedded.size()}')

        # Calculating Alignment Scores
        hidden_with_time_axis = hidden[0].permute(1, 0, 2)
        # print(f'hidden_with_time_axis = {hidden_with_time_axis.size()}')
        x = torch.tanh(self.fc_hidden(hidden_with_time_axis) + self.fc_encoder(encoder_outputs))
        # print(f'x = {x.size()}')
        # alignment_scores = self.V(x)
        # print(f'al scores = {alignment_scores.size()}')

        # Softmaxing alignment scores to get Attention weights
        attn_weights = F.softmax(self.V(x), dim=1)
        # print(f'at w = {attn_weights.size()}')

        # Multiplying the Attention weights with encoder outputs to get the context vector
        context_vector = torch.sum(attn_weights * encoder_outputs, dim=1)
        # print(f'context vector = {context_vector.size()}')

        # Concatenating context vector with embedded input word
        output = torch.cat((embedded, context_vector.unsqueeze(1)), -1)
        # print(f'concat = {output.size()}')
        # Passing the concatenated vector as input to the LSTM cell
        output, hidden = self.lstm(output, hidden)
        # print(f'output = {output.size()}')
        # Passing the LSTM output through a Linear layer acting as a classifier
        output = F.log_softmax(self.classifier(output.view(-1, output.size(2))), dim=1)
        return output, hidden, attn_weights


if __name__ == '__main__':
    src_del_field_name = 'src_del'
    src_add_field_name = 'src_add'
    tgt_field_name = 'tgt'
    src = SourceField(init_token='<sos>', eos_token='<eos>')
    tgt = TargetField(init_token='<sos>', eos_token='<eos>')  # init_token='<sos>', eos_token='<eos>'
    train = torchtext.data.TabularDataset(
        path='data/diffs/test/val.small.del_add_full.data', format='tsv',
        fields=[(src_del_field_name, src),
                (src_add_field_name, src),
                (tgt_field_name, tgt)],
    )
    src.build_vocab(train, max_size=50000)
    tgt.build_vocab(train, max_size=50000, min_freq=1)
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    print(f'src vocab = {len(src.vocab)}, tgt vocab = {len(tgt.vocab)}')
    print(output_vocab.stoi.keys())
    print(input_vocab.stoi)
    # Reading the English-German sentences pairs from the file
    with open("data/diffs/test/val.small.data", "r") as file:
        deu = [x for x in file.readlines()]
    en = []
    de = []
    for line in deu:
        en.append(line.split("\t")[0])
        de.append(line.split("\t")[1])

    # Setting the number of training sentences we'll use
    training_examples = 10
    # We'll be using the spaCy's English and German tokenizers
    spacy_en = English()
    spacy_de = German()

    en_words = Counter()
    de_words = Counter()
    en_inputs = []
    de_inputs = []

    # Tokenizing the English and German sentences and creating our word banks for both languages
    for i in tqdm(range(training_examples)):
        en_tokens = spacy_en(en[i])
        de_tokens = spacy_de(de[i])
        if len(en_tokens) == 0 or len(de_tokens) == 0:
            continue
        for token in en_tokens:
            en_words.update([token.text.lower()])
        en_inputs.append([token.text.lower() for token in en_tokens])
        for token in de_tokens:
            de_words.update([token.text.lower()])
        de_inputs.append([token.text.lower() for token in de_tokens])

    # Assigning an index to each word token, including the Start Of String(SOS), End Of String(EOS) and Unknown(UNK) tokens
    en_words = ['<sos>', '<ens>', '<unk>'] + sorted(en_words, key=en_words.get, reverse=True)
    en_w2i = {o: i for i, o in enumerate(en_words)}
    en_i2w = {i: o for i, o in enumerate(en_words)}
    de_words = ['<sos>', '<eos>', '<unk>'] + sorted(de_words, key=de_words.get, reverse=True)
    de_w2i = {o: i for i, o in enumerate(de_words)}
    de_i2w = {i: o for i, o in enumerate(de_words)}

    print(f'en_words = {len(en_words)}, de_words = {len(de_words)}')
    print(de_words)

    # Converting our English and German sentences to their token indexes
    for i in range(len(en_inputs)):
        en_sentence = en_inputs[i]
        de_sentence = de_inputs[i]
        en_inputs[i] = [en_w2i[word] for word in en_sentence]
        de_inputs[i] = [de_w2i[word] for word in de_sentence]

    hidden_size = 256
    encoder = EncoderLSTM(len(en_words), hidden_size).to(device)
    decoder = BahdanauDecoder(hidden_size, len(de_words)).to(device)

    lr = 0.001
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)

    EPOCHS = 1
    batch_size = 1
    teacher_forcing_prob = 1
    encoder.train()
    decoder.train()
    tk0 = tqdm(range(1, EPOCHS + 1), total=EPOCHS)
    batch_iterator = torchtext.data.BucketIterator(
        dataset=train,
        batch_size=batch_size,
        sort=False,
        repeat=False,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src_del),
        device=device)

    for epoch in tk0:
        batch_generator = batch_iterator.__iter__()
        avg_loss = 0.
        # tk1 = tqdm(enumerate(en_inputs), total=len(en_inputs), leave=False)
        # for i, sentence in tk1:
        for batch in batch_generator:
            # print(batch.data[0].src_del)
            print(batch.data[0].tgt)

            loss = 0.
            h = encoder.init_hidden(batch_size=batch_size)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            input_variables, input_lengths = getattr(batch, src_del_field_name)
            target_variables = getattr(batch, tgt_field_name)
            print(target_variables)
            # inp = torch.tensor(sentence).unsqueeze(0).to(device)
            # encoder_outputs, h = encoder(inp, h)
            # print(f'\ninput var = {input_variables.size()}, h_0 = {h[0].size()}, h_1 = {h[1].size()}')
            encoder_outputs, h = encoder(input_variables, h)
            # print(f'enc output= {encoder_outputs.size()}')
            # print(len(en_words), len(de_words))

            # First decoder input will be the SOS token
            decoder_input = torch.tensor([en_w2i['<sos>']] * batch_size, device=device)
            # First decoder hidden state will be last encoder hidden state
            decoder_hidden = h
            output = []
            teacher_forcing = True if random.random() < teacher_forcing_prob else False
            # print(f'target_variables = {target_variables.size()}')
            for ii in range(target_variables.size(1)):
                # print(f'decoder input = {decoder_input.size()}, encoder_outputs = {encoder_outputs.size()}')
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                                            decoder_hidden,
                                                                            encoder_outputs)
                # Get the index value of the word with the highest score from the decoder output
                # print(f'decoder output = {decoder_output.size()}')
                # print(f'tt = {target_variables}')
                # print(f'target_var = {target_variables[:, ii]}')
                top_value = decoder_output.argmax(1)
                # print(f'top index = {top_index}')
                output.append(top_value)
                # print(f'top value = {top_value}')
                # print(f'LOSS: decoder output = {decoder_output.size()}, tar var = {target_variables[:, ii].size()}')
                # print(f'LOSS: decoder_output = {decoder_output.view(-1, len(de_words)).size()},'
                #     f'target_vars = {target_variables[:, ii].view(-1).size()}')
                # loss += F.nll_loss(decoder_output.view(-1, len(de_words)), target_variables[:, ii].view(-1))
                loss += F.nll_loss(decoder_output, target_variables[:, ii])
                # print('after loss')
                if teacher_forcing:
                    decoder_input = target_variables[:, ii]  # Teacher forcing
                else:
                    decoder_input = top_value
                    if decoder_input.item() == '<eos>':
                        break
                # Calculate the loss of the prediction against the actual word
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            avg_loss += loss.item() / len(en_inputs)
        tk0.set_postfix(loss=avg_loss)

    #   # Save model after every epoch (Optional)
    # torch.save({"encoder":encoder.state_dict(),
    #             "decoder":decoder.state_dict(),
    #             "e_optimizer":encoder_optimizer.state_dict(),
    #             "d_optimizer":decoder_optimizer},
    #            "./model.pt")