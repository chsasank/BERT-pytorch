from torch import nn
from .utils import gelu
from .attention import Attention


class FeedForward(nn.Module):
    def __init__(self,
                 inp_size,
                 intermediate_size,
                 intermediate_act_fn,
                 hidden_dropout_prob):
        super(FeedForward, self).__init__()
        self.intermediate_layer = nn.Linear(inp_size, intermediate_size)
        self.intermediate_act_fn = intermediate_act_fn
        self.output_layer = nn.Linear(intermediate_size, inp_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, inp):
        out = self.intermediate_layer(inp)
        out = self.intermediate_act_fn(out)
        out = self.output_layer(out)
        out = self.dropout(out)

        return out


class TransformerLayer(nn.Module):
    def __init__(self,
                 hidden_size=768,
                 intermediate_size=3072,
                 intermediate_act_fn=gelu,
                 num_attention_heads=12,
                 attention_head_size=64,
                 attention_probs_dropout_prob=0.1,
                 hidden_dropout_prob=0.1,
                 initializer_range=0.02):
        super(TransformerLayer, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_act_fn = intermediate_act_fn

        self.attention = Attention(
            from_tensor_width=hidden_size,
            to_tensor_width=hidden_size,
            out_tensor_width=hidden_size,
            num_attention_heads=num_attention_heads,
            size_per_head=attention_head_size,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            initializer_range=initializer_range
        )
        self.feedforward = FeedForward(
            inp_size=hidden_size,
            intermediate_size=intermediate_size,
            intermediate_act_fn=intermediate_act_fn,
            hidden_dropout_prob=hidden_dropout_prob
        )

        self.layernorm_1 = nn.LayerNorm(hidden_size)
        self.layernorm_2 = nn.LayerNorm(hidden_size)

    def forward(self, from_tensor, to_tensor, attention_mask=None):
        assert from_tensor.shape[-1] == to_tensor.shape[-1] == self.hidden_size
        attention_output = self.attention(
            from_tensor, to_tensor, attention_mask)
        # residual + layer norm
        attention_output = self.layernorm_1(attention_output + from_tensor)

        layer_output = self.feedforward(attention_output)
        # residual + layer norm
        layer_output = self.layernorm_2(layer_output + attention_output)

        return layer_output


class Transformer(nn.Module):
    """Multi-headed, multi-layer Transformer from "Attention is All You Need".

    """

    def __init__(self,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 intermediate_act_fn=gelu,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 initializer_range=0.02):
        """
        Args:
            input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
            attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
              seq_length], with 1 for positions that can be attended to and 0 in
              positions that should not be.
            hidden_size: int. Hidden size of the Transformer.
            num_hidden_layers: int. Number of layers (blocks) in the Transformer.
            num_attention_heads: int. Number of attention heads in the Transformer.
            intermediate_size: int. The size of the "intermediate" (a.k.a., feed
              forward) layer.
            intermediate_act_fn: function. The non-linear activation function to apply
              to the output of the intermediate/feed-forward layer.
            hidden_dropout_prob: float. Dropout probability for the hidden layers.
            attention_probs_dropout_prob: float. Dropout probability of the attention
              probabilities.
            initializer_range: float. Range of the initializer (stddev of truncated
              normal).
            do_return_all_layers: Whether to also return all layers or just the final
              layer.
        """
        super(Transformer, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of"
                " attention heads (%d)" % (hidden_size, num_attention_heads))

        attention_head_size = int(hidden_size / num_attention_heads)

        self.transformer_layers = nn.ModuleList([
            TransformerLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                intermediate_act_fn=intermediate_act_fn,
                num_attention_heads=num_attention_heads,
                attention_head_size=attention_head_size,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                hidden_dropout_prob=hidden_dropout_prob,
                initializer_range=initializer_range)
            for i in range(num_hidden_layers)
        ])

    def forward(self, input_tensor, attention_mask=None,
                do_return_all_layers=False):
        """
        Args:
            input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
            attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
              seq_length], with 1 for positions that can be attended to and 0 in
              positions that should not be.
            do_return_all_layers: Whether to also return all layers or just the final
              layer.

        Returns:
            float Tensor of shape [batch_size, seq_length, hidden_size], the final
            hidden layer of the Transformer.

        Raises:
            ValueError: A Tensor shape or parameter is invalid.
        """
        assert len(input_tensor.shape) == 3
        batch_size, seq_length, input_width = input_tensor.shape

        # The Transformer performs sum residuals on all layers so the input
        # needs to be the same as the hidden size.
        if input_width != self.hidden_size:
            raise ValueError(
                "The width of the input tensor (%d) != hidden size (%d)" %
                (input_width, self.hidden_size))

        prev_output = input_tensor
        all_layer_outputs = []
        for i in range(self.num_hidden_layers):
            layer_output = self.transformer_layers[i](
                from_tensor=prev_output,
                to_tensor=prev_output,
                attention_mask=attention_mask)
            all_layer_outputs.append(layer_output)

        if do_return_all_layers:
            return all_layer_outputs
        else:
            return layer_output
