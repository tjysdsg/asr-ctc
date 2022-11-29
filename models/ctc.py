from lm import WordLM
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List


class CTC(nn.Module):
    def __init__(self, odim: int, idim: int):
        """Calculate CTC loss for the output of the encoder
        :param odim: output dimension, i.e. vocabulary size including the blank label
        :param idim: input dimension of ctc linear layer
        """
        super().__init__()
        self.projection = nn.Linear(idim, odim)
        self.ctc_loss = nn.CTCLoss(blank=0, reduction="sum", zero_infinity=True)

    def forward(self, hs, hlens, ys, ylens):
        """
        :param hs: output of the encoder, (N, T, eprojs)
        :param hlens: (N,)
        :param ys: padded target sequences
        :param ylens: (N,)
        """
        if not isinstance(ylens, torch.Tensor):
            ylens = torch.tensor(ylens, dtype=torch.long, device=hs.device)

        log_probs = F.log_softmax(self.projection(hs), dim=-1).transpose(
            0, 1
        )  # (T, N, odim)
        loss = self.ctc_loss(log_probs, ys, hlens, ylens)
        loss = loss / len(hlens)  # batch mean
        return loss

    def greedy_search(self, hs_pad):
        """Perform greedy search for the CTC.

        Args:
            hs_pad: (B, T, idim)
        Returns:
            results: list of lists of ints
        """
        # 1. apply linear projection and take argmax for each time step
        x = torch.argmax(self.projection(hs_pad), dim=-1)  # (B, T)
        x = x.detach().cpu().numpy()

        # 2. for each sample in the batch, merge repeated tokens and remove the blank token (index is 0)
        batch_size = x.shape[0]
        T = x.shape[1]

        results = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            prev = 0
            for t in range(T):
                token = x[i][t]
                if token != 0 and token != prev:
                    results[i].append(token)
                prev = token

        return results

    def beam_search(self, hs_pad, lens, beam_size: int, vocab: Dict[int, str], lm: WordLM):
        """
        Perform beam search for the CTC.

        Args:
            hs_pad: (B, T, idim)
            lens: (B)
            beam_size: int
            vocab: token index to its string
            lm: Language model
        Returns:
            results: list of lists of ints
        """
        # 1. apply linear projection and take argmax for each time step
        x = F.softmax(self.projection(hs_pad), dim=-1)  # (B, T, odim)

        # 2. beam search in parallel
        x = x.detach().cpu().numpy()
        batch_size = x.shape[0]
        results = [None for _ in range(batch_size)]

        # from concurrent.futures import ThreadPoolExecutor
        # with ThreadPoolExecutor() as executor:
        #     running_tasks = [
        #         executor.submit(lambda: self._beam_search_trellis(x[i], beam_size)) for i in range(batch_size)
        #     ]
        #     for i, running_task in enumerate(running_tasks):
        #         results[i] = running_task.result()

        logging.info(f'\nBeam search on batch with {batch_size} samples')
        for i in range(batch_size):
            results[i] = self._beam_search_trellis(x[i], lens[i], beam_size, vocab, lm)
        return results

    def _beam_search_trellis(
            self, x: np.ndarray, length: int, beam_size: int,
            vocab: Dict[int, str], lm: WordLM,
            space_char='▁', blank=0, min_threshold=0.001,
            alpha=0.3, beta=0.3,
    ):
        """
        https://arxiv.org/abs/1408.2873

        Slightly modified because the BPE has the space character at the beginning of the token
        """
        dim = x.shape[1]

        def prefix_to_words(pref: str):
            toks = [vocab[int(tok_i)] for tok_i in pref.split(' ') if tok_i != '']
            if len(toks):
                return ''.join(toks).split(space_char)
            else:
                return []

        hyps = ['']
        pb = defaultdict(int)  # prob_prefix_ends_with_blank
        pnb = defaultdict(int)  # prob_prefix_ends_with_non_blank
        pb[''] = 1
        pnb[''] = 0
        for t in range(length):
            candidate_idx = [i for i in range(dim) if x[t][i] > min_threshold]

            new_pb = defaultdict(int)
            new_pnb = defaultdict(int)
            for prefix in hyps:  # prefix is a space separated string containing token INDICES (easier for dict keys)
                L = prefix.split(' ')

                # TODO: batch run language model
                # curr_words = prefix_to_words(prefix)
                # sentences = [prefix_to_words(f'{prefix} {i}') if i != blank else curr_words for i in candidate_idx]
                # sentences = [' '.join(s) for s in sentences]
                # lm_probs = lm(sentences)
                # lm_probs = lm_probs[:, -1] ** alpha

                for ci, i in enumerate(candidate_idx):
                    P = x[t][i]

                    if i == blank:  # extend with a blank
                        new_pb[prefix] += P * (pb[prefix] + pnb[prefix])
                    else:  # extend with a non-blank
                        new_prefix = f'{prefix} {i}'

                        if L[-1] == str(i):  # repeating the last token
                            new_pnb[new_prefix] += P + pb[prefix]
                            new_pnb[prefix] += P * pb[prefix]
                        # extend with other non-blank token
                        else:
                            # TODO: language model
                            # token = vocab[i]
                            # lm_p = 1.0
                            # if space_char in token and token != space_char and lm_probs is not None:
                            #     lm_p = lm_probs[ci]
                            new_pnb[new_prefix] += P * (pb[prefix] + pnb[prefix])

                        # make use of discarded prefixes
                        if new_prefix not in hyps:
                            new_pb[new_prefix] += x[t][blank] * (pb[new_prefix] + pnb[new_prefix])
                            new_pnb[new_prefix] += P * pnb[new_prefix]

            pb = new_pb
            pnb = new_pnb

            # merge pnb and pb
            hyp2score = pb
            for k in pnb.keys():
                hyp2score[k] += pnb[k]

            # top k
            hyps = sorted(
                # TODO: hyp2score, key=lambda l: hyp2score[l] * len(prefix_to_words(l)) ** beta, reverse=True
                hyp2score, key=lambda l: hyp2score[l], reverse=True
            )[:beam_size]

        # clean
        result = []
        prev = blank
        tokens = [int(e) for e in hyps[0].strip(' ').split(' ')]
        for token in tokens:
            if token != blank and token != prev:
                result.append(token)
            prev = token
        return result


def test_beam_search():
    from train import load_configs
    configs = load_configs(["--out-dir=exp", "--tag=test"])

    batch_size = 3
    dim = len(configs.char_list)
    T = 5

    ctc = CTC(dim, dim).cuda()
    lm = WordLM().cuda()
    x = torch.rand((batch_size, T, dim)).cuda()
    res = ctc.beam_search(x, [T for _ in range(batch_size)], dim // 2, configs.char_list, lm)
    print(res)


if __name__ == '__main__':
    test_beam_search()
