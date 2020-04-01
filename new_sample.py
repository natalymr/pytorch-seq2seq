from pathlib import Path
from typing import Any

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from tqdm import tqdm
import random

from seq2seq.dataset import SourceField, TargetField

# if torch.cuda.is_available:
#     device = torch.device("cuda")
# else:
from results_analyzing.utils import run_perl_script_and_parse_result

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
        self.fc_encoder_del = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_encoder_add = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_encoder_common = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size))
        self.V_del = nn.Linear(self.hidden_size, 1)
        self.V_add = nn.Linear(self.hidden_size, 1)
        self.V_common = nn.Linear(self.hidden_size, 1)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.drop_prob)
        self.lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size, batch_first=True)
        self.classifier = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs, hidden, encoder_outputs_del, encoder_outputs_add):
        # print(f'DECODER')
        encoder_outputs_del = encoder_outputs_del.squeeze()
        encoder_outputs_add = encoder_outputs_add.squeeze()
        # print(f'D-forward = {inputs.size()},  h = {hidden[0].size()}, c={hidden[1].size()},'
        #       f'encoder output del = {encoder_outputs_del.size()}, encoder output add = {encoder_outputs_add.size()}')
        # Embed input words
        embedded = self.embedding(inputs[:, None])
        embedded = self.dropout(embedded)
        # print(f'embedded = {embedded.size()}, inputs = {inputs[:, None].size()}')

        # Calculating Alignment Scores for both encoders
        hidden_with_time_axis = hidden[0].permute(1, 0, 2)
        # print(f'hidden_with_time_axis = {hidden_with_time_axis.size()}')
        x_del = torch.tanh(self.fc_hidden(hidden_with_time_axis) + self.fc_encoder_del(encoder_outputs_del))
        x_add = torch.tanh(self.fc_hidden(hidden_with_time_axis) + self.fc_encoder_add(encoder_outputs_add))
        # print(f'x del = {x_del.size()}, x add = {x_add.size()}')

        # Softmaxing alignment scores to get Attention weights
        attn_weights_del = F.softmax(self.V_del(x_del), dim=1)
        attn_weights_add = F.softmax(self.V_add(x_add), dim=1)
        # print(f'at w del = {attn_weights_del.size()}, at w add = {attn_weights_add.size()}')

        # Multiplying the Attention weights with encoder outputs to get the context vector
        context_vector_del = torch.sum(attn_weights_del * encoder_outputs_del, dim=1)
        context_vector_add = torch.sum(attn_weights_add * encoder_outputs_add, dim=1)
        # print(f'context vector del = {context_vector_del.size()}, context vector add = {context_vector_add.size()}')
        context_vector_del = context_vector_del.unsqueeze(1)
        context_vector_add = context_vector_add.unsqueeze(1)
        # context_vector_del = context_vector_del.new_full(context_vector_del.size(), 1)
        # context_vector_add = context_vector_add.new_full(context_vector_add.size(), 2)
        # print(f'context vector del = {context_vector_del.size()}, context vector add = {context_vector_add.size()}')
        # COMMON
        # Calculating Alignment Common Scores
        context_vector_common = torch.cat((context_vector_del, context_vector_add), 1)
        # print(f'context vector concat = {context_vector_common.size()}')
        # print(context_vector_add)
        # print(context_vector_del)
        # print(context_vector_common)
        x_common = torch.tanh(self.fc_hidden(hidden_with_time_axis) + self.fc_encoder_common(context_vector_common))
        # print(f'x common = {x_common.size()}')

        # Softmaxing alignment scores to get Attention Common weights
        attn_weights = F.softmax(self.V_common(x_common), dim=1)
        # print(f'attn_weights common = {attn_weights.size()}')

        # Multiplying the Attention Common weights with context vectors to get the context Common vector
        context_vector = torch.sum(attn_weights * context_vector_common, dim=1)
        # print(f'context_vector common = {context_vector.size()}')

        # Concatenating context vector with embedded input word
        output = torch.cat((embedded, context_vector.unsqueeze(1)), -1)
        # print(f'output = {output.size()}')
        # print(f'output = {output}')
        # Passing the concatenated vector as input to the LSTM cell
        output, hidden = self.lstm(output, hidden)
        # print(f'output = {output.size()}')
        # print(f'output = {output}')
        # Passing the LSTM output through a Linear layer acting as a classifier
        output = F.log_softmax(self.classifier(output.view(-1, output.size(2))), dim=1)
        return output, hidden, attn_weights, attn_weights_del, attn_weights_add


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def forward(self, src, tgt, batch_size):



if __name__ == '__main__':
    perl_script_path = Path('../../../results_analyzing/bleu/multi-bleu.perl')
    specials = ['<sos>', '<eos>', '<unk>', '<pad>']
    src_del_field_name = 'src_del'
    src_add_field_name = 'src_add'
    tgt_field_name = 'tgt'
    src = SourceField(init_token='<sos>', eos_token='<eos>')
    tgt = TargetField(init_token='<sos>', eos_token='<eos>')  # init_token='<sos>', eos_token='<eos>'
    train = torchtext.data.TabularDataset(
        path='data/diffs/test/val.small.del_add.data', format='tsv',
        # path='../../../../new_data/processed_data/splitted_two_input_100/train_100.data', format='tsv',
        fields=[(src_del_field_name, src),
                (src_add_field_name, src),
                (tgt_field_name, tgt)],
    )
    src.build_vocab(train, max_size=50000)
    tgt.build_vocab(train, max_size=50000, min_freq=1)
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    # print(f'src vocab = {len(src.vocab)}, tgt vocab = {len(tgt.vocab)}')
    # print(output_vocab.stoi.keys())
    # print(input_vocab.stoi)
    # print(output_vocab.stoi)

    wandb.init(project="nmt-2.0-test")
    wandb.watch_called = False

    hidden_size = 2
    encoder = EncoderLSTM(len(src.vocab), hidden_size).to(device)
    decoder = BahdanauDecoder(hidden_size, len(tgt.vocab)).to(device)
    model = Seq2Seq(encoder, decoder)


    def get_model_size(model):
        def get_values_from_size(size_):
            result = tuple()
            for s in size_:
                result += (s,)
            return str(result)
        print(f'\n {model._get_name()}')
        print('| ======== Layer ======== | ======== Shape ======== | ======== #Params ======== |')
        total_num_params = 0
        for k in model.state_dict().keys():
            v = model.state_dict().__getitem__(k).size()
            total_num_params += v.numel()
            print("{:>25} {:>25} {:>27}".format(k, get_values_from_size(v), v.numel()))
        print('| ======================= | ======================= | ========================= |')
        print(f' Total number of params = {total_num_params}\n')

    # get_model_size(encoder)
    # get_model_size(decoder)

    lr = 0.001
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    EPOCHS = 1
    batch_size = 1
    teacher_forcing_prob = 1
    encoder.train()
    decoder.train()
    for name, sub_module in decoder.named_children():
        print(f'name = {name}, sm = {sub_module}')
    wandb.watch(decoder, log="all")

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
        tgt_seqs, pred_seqs = [], []
        # tk1 = tqdm(enumerate(en_inputs), total=len(en_inputs), leave=False)
        # for i, sentence in tk1:
        for batch in batch_generator:
            # print(batch.data[0].src_del)
            # print(batch.data[0].tgt)
            loss = 0.
            h_del = encoder.init_hidden(batch_size=batch_size)
            h_add = encoder.init_hidden(batch_size=batch_size)
            optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            input_variables_del, input_lengths_del = getattr(batch, src_del_field_name)
            input_variables_add, input_lengths_add = getattr(batch, src_add_field_name)
            target_variables = getattr(batch, tgt_field_name)

            encoder_outputs_del, h_del = encoder(input_variables_del, h_del, is_del=True)
            encoder_outputs_add, h_add = encoder(input_variables_add, h_add, is_del=False)

            # First decoder input will be the SOS token
            decoder_input = torch.tensor([output_vocab.stoi['<sos>']] * batch_size, device=device)
            # First decoder hidden state will be last encoder hidden state
            decoder_hidden = (h_del[0].add(h_add[0]), h_del[1].add(h_add[1]))
            output = []
            teacher_forcing = True if random.random() < teacher_forcing_prob else False
            teacher_forcing = True
            if not teacher_forcing:
                print(f'not TEACHER FORCHING')

            tgt_seqs.extend(
                ' '.join(batch.data[i].tgt)
                for i in range(len(batch))
            )

            for ii in range(target_variables.size(1)):
                # print(f'decoder input = {decoder_input.size()}, encoder_outputs = {encoder_outputs.size()}')
                decoder_output, decoder_hidden, decoder_attention, _, _ = decoder(decoder_input,
                                                                                  decoder_hidden,
                                                                                  encoder_outputs_del,
                                                                                  encoder_outputs_add)
                # Get the index value of the word with the highest score from the decoder output
                top_value = decoder_output.argmax(1)
                if ii == 0:
                    acc_tensor = top_value.reshape(batch_size, -1)
                else:
                    acc_tensor = torch.cat((acc_tensor, top_value.reshape(batch_size, -1)), 1)
                loss += F.nll_loss(decoder_output, target_variables[:, ii])
                if teacher_forcing:
                    decoder_input = target_variables[:, ii]  # Teacher forcing
                else:
                    decoder_input = top_value
                    # if decoder_input.item() == output_vocab.stoi['<eos>']:
                    #     break
                # Calculate the loss of the prediction against the actual word
            pred_seqs.extend(
                ' '.join(output_vocab.itos[tok] for tok in acc_tensor[i]
                         if output_vocab.itos[tok] not in specials)
                for i in range(len(batch))
            )

            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            avg_loss += loss.item() / len(batch)

        # compute bleu
        epoch_bleu = run_perl_script_and_parse_result('\n'.join(tgt_seqs),
                                                      '\n'.join(pred_seqs),
                                                      perl_script_path)
        wandb.log({'bleu': epoch_bleu.bleu, 'avg_loss': avg_loss})
        tk0.set_postfix(loss=avg_loss, train_bleu=epoch_bleu.bleu)
    wandb.save('model_nmt2.h5')

    #   # Save model after every epoch (Optional)
    # torch.save({"encoder":encoder.state_dict(),
    #             "decoder":decoder.state_dict(),
    #             "e_optimizer":encoder_optimizer.state_dict(),
    #             "d_optimizer":decoder_optimizer},
    #            "./model.pt")