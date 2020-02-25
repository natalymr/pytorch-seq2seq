from __future__ import print_function, division

from pathlib import Path

import torch
import torchtext


import seq2seq
from seq2seq.loss import NLLLoss

from results_analyzing.utils import run_perl_script_and_parse_result


# monkey-patching like a boss: Start
from torchtext.data import Batch
batch_old_init = Batch.__init__
def batch_new_init(self, data=None, dataset=None, device=None):
    self.data = data
    batch_old_init(self, data, dataset, device)

Batch.__init__ = batch_new_init
# monkey-patching like a boss: End


class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64):
        self.loss = loss
        self.batch_size = batch_size
        self.perl_script_path = Path('../../../../results_analyzing/bleu/multi-bleu.perl')

    def evaluate(self, model, data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()


        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)
        tgt_vocab = data.fields[seq2seq.tgt_field_name].vocab
        pad = tgt_vocab.stoi[data.fields[seq2seq.tgt_field_name].pad_token]
        tgt_seqs = []
        pred_seqs = []

        specials = ['<sos>', '<eos>', '<unk>', '<pad>']

        with torch.no_grad():
            for batch in batch_iterator:
                input_variables, input_lengths = getattr(batch, seq2seq.src_field_name)
                target_variables = getattr(batch, seq2seq.tgt_field_name)
                decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(), target_variables)

                # Evaluation
                seqlist = other['sequence']

                # get predicted and target sequences
                acc = other['sequence'][0]
                for i in range(1, len(other["sequence"])):
                    acc = torch.cat((acc, other['sequence'][i]), 1)
                # print(f'ACC = {acc}')
                # print(f'SEQ = {seqlist}')
                pred_seqs.extend(
                    ' '.join(tgt_vocab.itos[tok] for tok in acc[i]
                             if tgt_vocab.itos[tok] not in specials)
                    for i in range(len(batch))
                )
                tgt_seqs.extend(
                    ' '.join(batch.data[i].tgt)
                    for i in range(len(batch))
                )

                # print(f'pred = {pred_seqs}\n'
                #       f'tgt = {tgt_seqs}')

                for step, step_output in enumerate(decoder_outputs):

                    target = target_variables[:, step + 1]

                    loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

                    non_padding = target.ne(pad)
                    # print(seqlist[step].view(-1))
                    # print(f'target = {target}')
                    # print(data.fields[seq2seq.tgt_field_name].init_token)

                    correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
                    match += correct
                    total += non_padding.sum().item()

        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total
        with open('/Users/natalia.murycheva/PycharmProjects/gitCommitMessageCollector/NMT/natalymr/pytorch-seq2seq/pred_ref.txt', 'w') as f:
            for p, r in zip(pred_seqs, tgt_seqs):
                f.write(f'Ref: {r} ; Pred: {p}\n')
        common_bleu = run_perl_script_and_parse_result('\n'.join(tgt_seqs),
                                                       '\n'.join(pred_seqs),
                                                       self.perl_script_path)

        return loss.get_loss(), accuracy, common_bleu
