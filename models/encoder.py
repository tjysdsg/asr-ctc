import torch
from torch import nn
from typing import Tuple

from models.layers import (
    LayerNorm,
    Conv2dSubsampling,
    PositionwiseFeedForward,
    MultiHeadedAttention,
)
from utils import make_pad_mask, repeat


class EncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        dropout_rate (float): Dropout rate.
    """

    def __init__(
            self,
            size,
            self_attn,
            feed_forward,
            dropout_rate,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size

    def forward(self, x, mask):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, 1, time).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, 1, time).

        """
        # attention with residual connection
        # mask is used in the attention module
        # x -> norm1 -> att -> dropout -> + -> x
        # |_______________________________|
        x_norm = self.norm1(x)
        x = x + self.dropout(
            self.self_attn(x_norm, x_norm, x_norm, mask)
        )

        # feed-forward network with residual connection
        # x -> norm2 -> ffn -> dropout -> + -> x
        # |_______________________________|
        x = x + self.dropout(self.feed_forward(self.norm2(x)))

        return x, mask


class TransformerEncoder(torch.nn.Module):
    """Transformer encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        positional_dropout_rate: dropout rate after adding positional encoding
    """

    def __init__(
            self,
            input_size: int,
            output_size: int = 256,
            attention_heads: int = 4,
            linear_units: int = 2048,
            num_blocks: int = 6,
            dropout_rate: float = 0.1,
            positional_dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.0,
    ):
        super().__init__()
        self._output_size = output_size

        self.embed = Conv2dSubsampling(input_size, output_size, positional_dropout_rate)

        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
        )

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                MultiHeadedAttention(
                    attention_heads, output_size, attention_dropout_rate
                ),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
            ),
        )
        self.after_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(
            self,
            xs_pad: torch.Tensor,
            ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
        Returns:
            position embedded tensor and mask
        """
        # prepare masks
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        # apply convolutional subsampling
        xs_pad, masks = self.embed(xs_pad, masks)

        # forward encoder layers
        xs_pad, masks = self.encoders(xs_pad, masks)

        # apply another layer norm at the end
        xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        return xs_pad, olens


def test_transformer_encoder():
    import random
    batch_size = 16
    input_size = 32
    seq_len = 64
    encoder = TransformerEncoder(32)

    x = torch.rand((batch_size, seq_len, input_size))
    ilens = torch.tensor([random.randint(10, seq_len) for _ in range(batch_size)])
    res, olens = encoder(x, ilens)
    print(res.shape)
    print(olens)
