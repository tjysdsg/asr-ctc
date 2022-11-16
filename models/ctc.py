from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging


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

    def beam_search(self, hs_pad, lens, beam_size: int):
        """
        Perform beam search for the CTC.

        Args:
            hs_pad: (B, T, idim)
            lens: (B)
            beam_size: int
        Returns:
            results: list of lists of ints
        """
        # 1. apply linear projection and take argmax for each time step
        x = F.log_softmax(self.projection(hs_pad), dim=-1)  # (B, T, odim)

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
            results[i] = self._beam_search_trellis(x[i], lens[i], beam_size)
        return results

    def _beam_search_trellis(self, x: np.ndarray, length: int, beam_size: int, blank=0):
        # TODO: language model
        dim = x.shape[1]

        # running_hyps[b] = [
        #   (seq, score)  # hyp 1
        #   hyp_2
        #   ...
        #   hyp_n  # n = beam_size
        # ]
        running_hyps = [([0], 0.0)]
        for t in range(length):
            new_hyps = []
            for hyp in running_hyps:  # type: Tuple[list, float]
                for i in range(dim):
                    seq = hyp[0] + [i]  # copy
                    new_hyps.append((seq, hyp[1] + x[t][i]))

            running_hyps = sorted(new_hyps, key=lambda h: h[1], reverse=True)[:beam_size]

        # clean
        result = []
        prev = blank
        for token in running_hyps[0][0]:
            if token != blank and token != prev:
                result.append(token)
            prev = token
        return result


def test_beam_search():
    batch_size = 3
    dim = 10
    T = 5

    ctc = CTC(dim, dim)
    x = torch.rand((batch_size, T, dim))
    res = ctc.beam_search(x, [T for _ in range(batch_size)], dim // 2)
    print(res)
