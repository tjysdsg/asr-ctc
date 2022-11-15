import torch
import torch.nn as nn
import torch.nn.functional as F


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
