import torch
from torch import nn
import math
from .utils import reshape_to_matrix


class Attention(nn.Module):
    """Multi-headed attention."""

    def __init__(self,
                 from_tensor_width=768,
                 to_tensor_width=768,
                 out_tensor_width=768,
                 num_attention_heads=1,
                 size_per_head=512,
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 attention_probs_dropout_prob=0.0,
                 hidden_dropout_prob=0.1,
                 initializer_range=0.02):
        """
        This is an implementation of multi-headed attention based on "Attention
        is all you Need". If `from_tensor` and `to_tensor` are the same, then
        this is self-attention. Each timestep in `from_tensor` attends to the
        corresponding sequence in `to_tensor`, and returns a fixed-with vector.

        This function first projects `from_tensor` into a "query" tensor and
        `to_tensor` into "key" and "value" tensors. These are (effectively) a list
        of tensors of length `num_attention_heads`, where each tensor is of shape
        [batch_size, seq_length, size_per_head].

        Then, the query and key tensors are dot-producted and scaled. These are
        softmaxed to obtain attention probabilities. The value tensors are then
        interpolated by these probabilities, then concatenated back to a single
        tensor and returned.

        In practice, the multi-headed attention are done with transposes and
        reshapes rather than actual separate tensors.

        Args:
            from_tensor_width: int. Width of from_tensor, to_tensor and output.
            to_tensor_width: int. Width of from_tensor, to_tensor and output.
            out_tensor_width: int. Width of from_tensor, to_tensor and output.
            num_attention_heads: int. Number of attention heads.
            size_per_head: int. Size of each attention head.
            query_act: (optional) Activation function for the query transform.
            key_act: (optional) Activation function for the key transform.
            value_act: (optional) Activation function for the value transform.
            attention_probs_dropout_prob: (optional) float. Dropout probability of the
              attention probabilities.
            initializer_range: float. Range of the weight initializer.
        """
        super(Attention, self).__init__()
        self.from_tensor_width = from_tensor_width
        self.to_tensor_width = to_tensor_width
        self.out_tensor_width = out_tensor_width
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.query_act = query_act
        self.key_act = key_act
        self.value_act = value_act
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range

        self.query_layer = nn.Linear(
            from_tensor_width, num_attention_heads * size_per_head)
        self.key_layer = nn.Linear(
            to_tensor_width, num_attention_heads * size_per_head)
        self.value_layer = nn.Linear(
            to_tensor_width, num_attention_heads * size_per_head)

        self.attention_probs_dropout = nn.Dropout(
            attention_probs_dropout_prob)

        self.output_layer = nn.Linear(
            num_attention_heads * size_per_head, out_tensor_width)
        self.output_droput = nn.Dropout(hidden_dropout_prob)

    def _transpose_for_scores(self, input_tensor, batch_size, seq_length):
        output_tensor = torch.reshape(
            input_tensor, [batch_size, seq_length, self.num_attention_heads,
                           self.size_per_head])

        output_tensor = output_tensor.transpose(1, 2)
        return output_tensor

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`
    def project_tensors(self, from_tensor, to_tensor):
        """Project from and to tensors to queries, keys and values."""
        batch_size, from_seq_length = from_tensor.shape[0:2]
        to_seq_length = to_tensor.shape[1]

        from_tensor_2d = reshape_to_matrix(from_tensor)
        to_tensor_2d = reshape_to_matrix(to_tensor)

        queries = self.query_layer(from_tensor_2d)  # [B*F, N*H]
        if self.query_act:
            queries = self.query_act(queries)

        keys = self.key_layer(to_tensor_2d)         # [B*T, N*H]
        if self.key_act:
            keys = self.key_act(keys)

        values = self.value_layer(to_tensor_2d)     # [B*T, N*H]
        if self.value_act:
            values = self.value_act(values)

        # `queries` = [B, N, F, H]
        queries = self._transpose_for_scores(
            queries, batch_size, from_seq_length)

        # `keys` = [B, N, F, H]
        keys = self._transpose_for_scores(
            keys, batch_size, to_seq_length)

        # `values` = [B, N, T, H]
        values = self._transpose_for_scores(
            keys, batch_size, to_seq_length)

        return queries, keys, values

    def scaled_dot_product_attention(self, queries, keys, values,
                                     attention_mask=None,
                                     do_return_2d_tensor=False):
        """Multi headed attention.

        section 3.2.1 in the transformer paper
        """
        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        # `attention_scores` = [B, N, F, T]
        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(float(self.size_per_head))

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)  # [B, 1, F, T]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            adder = (1.0 - attention_mask.float()) * -10000.0

            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_scores += adder

        # Normalize the attention scores to probabilities.
        # `attention_probs` = [B, N, F, T]
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_probs_dropout(attention_probs)

        contexts = torch.matmul(attention_probs, values)  # [B, N, F, H]
        contexts = contexts.transpose(1, 2)  # [B, F, N, H]

        if do_return_2d_tensor:
            B, F, N, H = contexts.shape
            # `contexts` = [B*F, N*H]
            contexts = torch.reshape(contexts, [B * F, N * H])
        else:
            # `contexts` = [B, F, N*H]
            contexts = torch.reshape(contexts, [*contexts.shape[:2], -1])

        return contexts

    def forward(self, from_tensor, to_tensor, attention_mask=None):
        """
        Project tensors, apply attention and project output.

        Args:
            from_tensor: float Tensor of shape [batch_size, from_seq_length,
                from_tensor_width].
            to_tensor: float Tensor of shape [batch_size, to_seq_length,
                to_tensor_width].
            attention_mask: (optional) int32 Tensor of shape [batch_size,
              from_seq_length, to_seq_length]. The values should be 1 or 0. The
              attention scores will effectively be set to -infinity for any positions in
              the mask that are 0, and will be unchanged for positions that are 1.

        Returns:
            float Tensor of shape [batch_size, from_seq_length,
              num_attention_heads * size_per_head].

        Raises:
            ValueError: Any of the arguments or tensor shapes are invalid.
        """
        from_shape = from_tensor.shape
        to_shape = to_tensor.shape

        try:
            assert len(from_shape) == 3 and len(to_shape) == 3
            assert (from_shape[-1] == self.from_tensor_width and
                    to_shape[-1] == self.to_tensor_width), \
                'Unexpected number of features'

            assert from_shape[0] == to_shape[0], 'Batch sizes do not match'
        except AssertionError:
            raise ValueError(
                'Invalid argument sizes. from_tensor: {} to_tensor: {}'.format(
                    from_shape, to_shape))

        queries, keys, values = self.project_tensors(from_tensor, to_tensor)
        attention_output = self.scaled_dot_product_attention(
            queries, keys, values, attention_mask, do_return_2d_tensor=True)
        attention_output = self.output_layer(attention_output)
        attention_output = self.output_droput(attention_output)
        attention_output = torch.reshape(
            attention_output, [*from_tensor.shape[:-1], -1])

        return attention_output
