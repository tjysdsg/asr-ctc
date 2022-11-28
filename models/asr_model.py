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
            cnn_module_kernel=params.cnn_module_kernel,
            dropout_rate=params.edropout,
            positional_dropout_rate=params.edropout,
            attention_dropout_rate=params.edropout,
        )
        self.ctc = CTC(odim=params.odim, idim=params.hidden_dim)
        self.stat_calculator = StatsCalculator(params)
        self.beam = params.beam_size

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
        # 1. Encoder forward (CNN + Transformer)
        xs, xlens = self.encoder(xs, xlens)

        # 2. Compute CTC Loss
        loss = self.ctc(xs, xlens, ys_ref, ylen)

        # 3. Compute stats by calling `self.stat_calculator.compute_wer`
        wer = self.stat_calculator.compute_wer(self.ctc.greedy_search(xs), ys_ref)

        return loss, wer

    def decode_greedy(self, xs, xlens):
        """Perform Greedy Decoding using trained ASRModel

        :params torch.Tensor xs- Speech feature input
        :params list xlens- Lengths of unpadded feature sequences
        """
        # 1. forward encoder
        xs, xlens = self.encoder(xs, xlens)

        # 2. get the predictions by calling `self.ctc.greedy_search`
        predictions = self.ctc.greedy_search(xs)

        return predictions

    def decode_beam(self, xs, xlens):
        xs, xlens = self.encoder(xs, xlens)
        predictions = self.ctc.beam_search(xs, xlens, self.beam)
        return predictions


def test_asr_forward():
    import random
    from train import load_configs
    configs = load_configs(["--out-dir=exp", "--tag=test"])

    batch_size = 16
    input_size = configs.idim
    x_max_len = 256
    y_max_len = 8
    asr = ASRModel(configs)

    x = torch.rand((batch_size, x_max_len, input_size))
    xlens = torch.tensor([random.randint(x_max_len - 50, x_max_len) for _ in range(batch_size)])

    y = torch.randint(0, 100, (batch_size, y_max_len))
    ylens = torch.tensor([y_max_len for _ in range(batch_size)])

    loss, wer = asr(x, xlens, y, ylens)
    print('loss:', loss)
    print('wer:', wer)
