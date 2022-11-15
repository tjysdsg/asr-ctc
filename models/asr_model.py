import torch
import configargparse

from models.encoder import TransformerEncoder
from models.ctc import CTC
from utils import StatsCalculator


class ASRModel(torch.nn.Module):
    def __init__(self, params: configargparse.Namespace):
        """E2E ASR model implementation.

        Args:
            params: The training options
        """
        super().__init__()

        self.ignore_id = params.text_pad

        self.encoder = TransformerEncoder(
            input_size=params.idim,
            output_size=params.hidden_dim,
            attention_heads=params.attention_heads,
            linear_units=params.linear_units,
            num_blocks=params.eblocks,
            dropout_rate=params.edropout,
            positional_dropout_rate=params.edropout,
            attention_dropout_rate=params.edropout,
        )
        self.ctc = CTC(odim=params.odim, idim=params.hidden_dim)
        self.stat_calculator = StatsCalculator(params)

    def forward(
        self,
        xs,
        xlens,
        ys_ref,
        ylen,
    ):
        """Forward propogation for ASRModel

        :params torch.Tensor xs- Speech feature input
        :params list xlens- Lengths of unpadded feature sequences
        :params torch.LongTensor ys_ref- Padded Text Tokens
        :params list ylen- Lengths of unpadded text sequences
        """
        # TODO: implement forward of the ASR model

        # 1. Encoder forward (CNN + Transformer)

        # 2. Compute CTC Loss

        # 3. Compute stats by calling `self.stat_calculator.compute_wer`

        return loss, wer

    def decode_greedy(self, xs, xlens):
        """Perform Greedy Decoding using trained ASRModel

        :params torch.Tensor xs- Speech feature input
        :params list xlens- Lengths of unpadded feature sequences
        """
        # TODO: implement CTC greedy decoding for the ASR model

        # 1. forward encoder

        # 2. get the predictions by calling `self.ctc.greedy_search`

        return predictions
