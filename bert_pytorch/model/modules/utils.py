import torch
import math
import six
from torch.nn import functional as F


# https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `F.relu`.

    Args:
    activation_string: String name of the activation function.

    Returns:
        A Python function corresponding to the activation function. If
        `activation_string` is None, empty, or "linear", this will return None.
        If `activation_string` is not a string, it will return `activation_string`.

    Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
    """

    # We assume that anything that's not a string is already an activation
    # function, so we just return it.

    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return F.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return torch.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.ndimension()
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = torch.reshape(input_tensor, [-1, width])
    return output_tensor


def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.

    Args:
        from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
        to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
        float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """

    assert len(from_tensor.shape) in [2, 3]
    batch_size, from_seq_length = from_tensor.shape[:2]

    assert len(to_mask.shape) == 2 and to_mask.shape[0] == batch_size

    to_mask = to_mask.unsqueeze(1).float()

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]

    broadcast_ones = torch.ones([batch_size, from_seq_length, 1])

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask

    return mask
