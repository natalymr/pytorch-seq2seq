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

import sys
sys.path.append('/home/ubuntu/gcm')

from seq2seq.dataset import SourceField, TargetField
from results_analyzing.utils import run_perl_script_and_parse_result, get_nltk_bleu_score_for_corpora

# monkey-patching like a boss: Start
from torchtext.data import Batch
batch_old_init = Batch.__init__
def batch_new_init(self, data=None, dataset=None, device=None):
    self.data = data
    batch_old_init(self, data, dataset, device)

Batch.__init__ = batch_new_init
# monkey-patching like a boss: End

if torch.cuda.is_available:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


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


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, lstm_drop_prob=0., embed_drop_prob=0.):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.dropout = nn.Dropout(embed_drop_prob)
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm_del = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=lstm_drop_prob, batch_first=True)
        self.lstm_add = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=lstm_drop_prob, batch_first=True)

    def forward(self, inputs, hidden, is_del: bool):
        # Embed input words
        embedded = self.dropout(self.embedding(inputs))
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
    def __init__(self, hidden_size, output_size, n_layers=1, embed_drop_prob=0.1, lstm_drop_prob=0.1):
        super(BahdanauDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        self.fc_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_encoder_del = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_encoder_add = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_encoder_common = nn.Linear(self.hidden_size, self.hidden_size)
        self.V_del = nn.Linear(self.hidden_size, 1)
        self.V_add = nn.Linear(self.hidden_size, 1)
        self.V_common = nn.Linear(self.hidden_size, 1)
        self.dropout = nn.Dropout(embed_drop_prob)
        self.lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size, batch_first=True, dropout=lstm_drop_prob)
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


class Seq2SeqTwoInput(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqTwoInput, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def forward(self, src_del, src_add, tgt, batch_size):
        h_del = encoder.init_hidden(batch_size=batch_size)
        h_add = encoder.init_hidden(batch_size=batch_size)

        encoder_outputs_del, h_del = encoder(src_del, h_del, is_del=True)
        encoder_outputs_add, h_add = encoder(src_add, h_add, is_del=False)

        # First decoder input will be the SOS token
        decoder_input = torch.tensor([output_vocab.stoi['<sos>']] * batch_size, device=device)
        # First decoder hidden state will be last encoder hidden state
        decoder_hidden = (h_del[0].add(h_add[0]), h_del[1].add(h_add[1]))
        tgt_len = tgt.size(1)
        output_vocab_size = decoder.output_size
        output = torch.zeros(tgt_len, batch_size, output_vocab_size).to(device)
        teacher_forcing = True if random.random() < teacher_forcing_prob else False
        teacher_forcing = True
        if not teacher_forcing:
            print(f'not TEACHER FORCHING')

        for ii in range(tgt_len):
            decoder_output, decoder_hidden, decoder_attention, _, _ = decoder(decoder_input,
                                                                              decoder_hidden,
                                                                              encoder_outputs_del,
                                                                              encoder_outputs_add)
            # Get the index value of the word with the highest score from the decoder output
            top_value = decoder_output.argmax(1)
            output[ii] = decoder_output
            if ii == 0:
                acc_tensor = top_value.reshape(batch_size, -1)
            else:
                acc_tensor = torch.cat((acc_tensor, top_value.reshape(batch_size, -1)), 1)

            if teacher_forcing:
                decoder_input = tgt[:, ii]  # Teacher forcing
            else:
                decoder_input = top_value

        return output, acc_tensor


def train(model, iterator, optimizer, loss_f, break_after, ignore_index_loss):
    model.train()
    print(f'we need to stop after {break_after}')
    epoch_loss = 0.
    acc_loss = 0.
    tgt_seqs, pred_seqs = [], []
    batch_generator = iterator.__iter__()
    optimizer.zero_grad()

    for i, batch in enumerate(batch_generator):
        input_variables_del, _ = getattr(batch, src_del_field_name)
        input_variables_add, _ = getattr(batch, src_add_field_name)
        target_variables = getattr(batch, tgt_field_name)

        input_variables_del.to(device)
        input_variables_add.to(device)
        target_variables.to(device)

        output, pred_acc_tensor = model(input_variables_del, input_variables_add, target_variables, batch_size)
        # Calculate the loss of the prediction against the actual word
        output_dim = output.shape[-1]
        loss = loss_f(output.view(-1, output_dim), target_variables.view(-1), ignore_index=ignore_index_loss)
        # loss = loss_f(output.squeeze(1), target_variables.squeeze(0))

        tgt_seqs.extend(
            ' '.join(batch.data[j].tgt)
            for j in range(len(batch))
        )
        pred_seqs.extend(
            ' '.join(output_vocab.itos[tok] for tok in pred_acc_tensor[j]
                    # if output_vocab.itos[tok] not in specials)
            )
               for j in range(len(batch))
        )

        if i % 10 == 0:
            loss = 0.9 * acc_loss + 0.1 * loss
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            # for p in model.parameters():
            #     p.data.add_(-lr, p.grad.data)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            acc_loss = 0.
        else:
            acc_loss += loss
        epoch_loss += loss.item() / len(batch)
        if i  == break_after:
            print(f'Stop after {i}')
            iterator.init_epoch()
            break 

    # compute bleu
    #epoch_bleu = run_perl_script_and_parse_result('\n'.join(tgt_seqs),
    #                                              '\n'.join(pred_seqs),
    #                                              perl_script_path)

    nltk_bleu = get_nltk_bleu_score_for_corpora(tgt_seqs, pred_seqs)
    print(f'bleu train = {nltk_bleu}') 
    for p, t in zip(pred_seqs[:20], tgt_seqs[:20]):
        print(f'{p} - {t}')
    #print("TARGET = {}\nPREDICTED = {}".format(', '.join(tgt_seqs), ', '.join(pred_seqs)))
    return epoch_loss, nltk_bleu


def write_to_file(output, seqs):
    with open(output, 'w') as f:
        for seq in seqs:
            f.write(f'{seq}\n')


def evaluate(model, iterator, loss_f, ignore_index_loss, is_test=False):
    model.eval()

    epoch_loss = 0.
    tgt_seqs, pred_seqs = [], []
    batch_generator = iterator.__iter__()

    with torch.no_grad():
        for batch in batch_generator:
            input_variables_del, _ = getattr(batch, src_del_field_name)
            input_variables_add, _ = getattr(batch, src_add_field_name)
            target_variables = getattr(batch, tgt_field_name)

            input_variables_del.to(device)
            input_variables_add.to(device)
            target_variables.to(device)

            output, pred_acc_tensor = model(input_variables_del, input_variables_add, target_variables, batch_size)

            # Calculate the loss of the prediction against the actual word
            output_dim = output.shape[-1]
            loss = loss_f(output.view(-1, output_dim), target_variables.view(-1), ignore_index=ignore_index_loss)

            tgt_seqs.extend(
                ' '.join(batch.data[i].tgt)
                for i in range(len(batch))
            )
            pred_seqs.extend(
                ' '.join(output_vocab.itos[tok] for tok in pred_acc_tensor[i]
                         if output_vocab.itos[tok] not in specials)
                for i in range(len(batch))
            )

            epoch_loss += loss.item() / len(batch)

        # compute bleu
#        epoch_bleu = run_perl_script_and_parse_result('\n'.join(tgt_seqs),
#                                                      '\n'.join(pred_seqs),
#                                                      perl_script_path)
        nltk_bleu = get_nltk_bleu_score_for_corpora(tgt_seqs, pred_seqs)
        if is_test:
            write_to_file(Path(f'results_nmt2_tokens_100_2/pred_{nltk_bleu}.txt'), pred_seqs)
            write_to_file(Path(f'results_nmt2_tokens_100_2/ref.txt'), tgt_seqs)
 #       print("TARGET = {}\nPREDICTED = {}".format(', '.join(tgt_seqs), ', '.join(pred_seqs)))
    return epoch_loss, nltk_bleu


if __name__ == '__main__':
    perl_script_path = Path('../../../results_analyzing/bleu/multi-bleu.perl')
    specials = ['<sos>', '<eos>', '<unk>', '<pad>']
    src_del_field_name = 'src_del'
    src_add_field_name = 'src_add'
    tgt_field_name = 'tgt'
    src = SourceField(init_token='<sos>', eos_token='<eos>')
    tgt = TargetField(init_token='<sos>', eos_token='<eos>')  # init_token='<sos>', eos_token='<eos>'
    train_data = torchtext.data.TabularDataset(
        # path='data/diffs/test/val.small.del_add.data', format='tsv',
        # path='data/splitted_two_input_100/train_100_10.data', format='tsv',
         path='data/splitted_two_input_100/train_100_47500.data', format='tsv',
        # path='data/splitted_two_input_200/train_200_83000.data', format='tsv',
        # path='../../../../new_data/processed_data/splitted_two_input_100/train_100.data', format='tsv',
        fields=[(src_del_field_name, src),
                (src_add_field_name, src),
                (tgt_field_name, tgt)],
    )

    src.build_vocab(train_data, max_size=50000)
    tgt.build_vocab(train_data, max_size=50000, min_freq=1)
    input_vocab = src.vocab
    output_vocab = tgt.vocab
    pad_index = output_vocab.stoi['<pad>']

    test_data = torchtext.data.TabularDataset(
        # path='data/diffs/test/val.small.del_add.data', format='tsv',
        path='data/splitted_two_input_100/test_100_19000.data', format='tsv',
        # path='data/splitted_two_input_200/test_200_17000.data', format='tsv',
        # path='data/splitted_two_input_100/test_100.data', format='tsv',
        # path='../../../../new_data/processed_data/splitted_two_input_100/train_100.data', format='tsv',
        fields=[(src_del_field_name, src),
                (src_add_field_name, src),
                (tgt_field_name, tgt)],
    )
    val_data = torchtext.data.TabularDataset(
        path='data/splitted_two_input_100/test_100_19000.data', format='tsv',
        # path='data/splitted_two_input_200/val_200_16500.data', format='tsv',
        fields=[(src_del_field_name, src),
                (src_add_field_name, src),
                (tgt_field_name, tgt)],
    )

    # print(f'src vocab = {len(src.vocab)}, tgt vocab = {len(tgt.vocab)}')
    # print(output_vocab.stoi.keys())
    # print(input_vocab.stoi)
    # print(output_vocab.stoi)

    wandb.init(entity='natalymr', project="nmt-2.0-test")
    wandb.watch_called = False

    hidden_size = 300
    encoder = EncoderLSTM(len(src.vocab), hidden_size, embed_drop_prob=0.2, lstm_drop_prob=0.2).to(device)
    decoder = BahdanauDecoder(hidden_size, len(tgt.vocab)).to(device)
    model = Seq2SeqTwoInput(encoder, decoder).cuda()

    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                # nn.init.normal_(param.data, mean=0, std=0.01)
                nn.init.xavier_normal_(param.data)
                # nn.init.xavier_uniform_(param.data)
            else:
                nn.init.constant_(param.data, 0)

    model.apply(init_weights)
    get_model_size(model)
    print(f'src vocab = {len(src.vocab)}, tgt vocab = {len(tgt.vocab)}')

    optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    EPOCHS = 100
    batch_size = 25
    BS = 250
    test_every_epoch = 1
    teacher_forcing_prob = 1

    wandb.watch(decoder, log="all")

    tk0 = tqdm(range(1, EPOCHS + 1), total=EPOCHS)
    batch_iterator = torchtext.data.BucketIterator(
        dataset=train_data,
        batch_size=batch_size,
        sort=True,
        repeat=False,
        shuffle=False,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src_del),
        device=device)

    test_batch_iterator = torchtext.data.BucketIterator(
        dataset=test_data,
        batch_size=batch_size,
        sort=True,
        repeat=False,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src_del),
        device=device)

    val_batch_iterator = torchtext.data.BucketIterator(
        dataset=val_data,
        batch_size=batch_size,
        sort=True,
        repeat=False,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src_del),
        device=device)

    start_to_add_data = False
    number_of_commits_in_train = 47900 
    break_after_batch = int(number_of_commits_in_train / batch_size)   
    for epoch in tk0:
        epoch_loss, epoch_bleu = train(model, batch_iterator, optimizer, F.nll_loss, break_after_batch, ignore_index_loss=pad_index)
        
        if epoch % test_every_epoch == 0:
            test_loss, test_bleu = evaluate(model, test_batch_iterator, F.nll_loss, is_test=True, ignore_index_loss=pad_index)
            val_loss, val_bleu = evaluate(model, val_batch_iterator, F.nll_loss, ignore_index_loss=pad_index)
            print(f'test bleu = {test_bleu}')
            if test_bleu > 1.5:
                start_to_add_data = True
            wandb.log({'train_bleu': epoch_bleu,
                       'train_loss': epoch_loss,
                       'test_loss': test_loss,
                       'test_bleu': test_bleu,
                       'val_loss': val_loss,
                       'val_bleu': val_bleu})
            tk0.set_postfix(train_loss=epoch_loss, train_bleu=epoch_bleu,
                            test_loss=test_loss, test_bleu=test_bleu,
                            val_loss=val_loss, val_bleu=val_bleu)
        else:
            wandb.log({'train_bleu': epoch_bleu, 'train_loss': epoch_loss})
            tk0.set_postfix(loss=epoch_loss, train_bleu=epoch_bleu)
        if epoch % 10 == 0:
            torch.save({"encoder": encoder.state_dict(),
                 "decoder": decoder.state_dict(),
                 "optimizer": optimizer.state_dict()},
                "./model_nmt.pt") 
        if start_to_add_data:
            number_of_commits_in_train += 1000
            break_after_batch = int(number_of_commits_in_train / BS)
            
        scheduler.step()
    wandb.save('model_nmt2.h5')

    #   # Save model after every epoch (Optional)
    # torch.save({"encoder":encoder.state_dict(),
    #             "decoder":decoder.state_dict(),
    #             "e_optimizer":encoder_optimizer.state_dict(),
    #             "d_optimizer":decoder_optimizer},
    #            "./model.pt")
