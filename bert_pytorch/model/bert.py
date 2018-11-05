import torch
import torch.nn as nn
import copy

from .modules.transformer import Transformer
from .modules.embedding import BERTEmbedding
from .modules import utils


class BertConfig(object):
    """Configuration for `BertModel`."""

    def __init__(self, vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
          vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
          hidden_size: Size of the encoder layers and the pooler layer.
          num_hidden_layers: Number of hidden layers in the Transformer encoder.
          num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder.
          intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
          hidden_act: The non-linear activation function (function or string) in the
            encoder and pooler.
          hidden_dropout_prob: The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
          attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
          max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
          type_vocab_size: The vocabulary size of the `token_type_ids` passed into
            `BertModel`.
          initializer_range: The stdev of the truncated_normal_initializer for
            initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """
    def __init__(self, config):
        super(BERT, self).__init__()
        config = copy.deepcopy(config)

        self.embeddings = BERTEmbedding(
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            use_token_type=True,
            token_type_vocab_size=config.type_vocab_size,
            use_position_embeddings=True,
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)

        self.encoder = Transformer(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=utils.get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range
        )

    def forward(self, input_ids, input_mask=None, token_type_ids=None):
        assert len(input_ids.shape) == 2
        batch_size, seq_length = input_ids.shape

        if input_mask is None:
            input_mask = torch.ones([batch_size, seq_length], dtype=torch.long)

        attention_mask = utils.create_attention_mask_from_input_mask(
            input_ids, input_mask)

        embedding_output = self.embeddings(input_ids)
        all_encoder_layers = self.encoder(embedding_output, attention_mask,
                                          do_return_all_layers=True)

        return all_encoder_layers
