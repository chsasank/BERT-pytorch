import torch.nn as nn


class BERTEmbedding(nn.Module):
    """Embedding for BERT model.

    Equivalent functions: modeling.embedding_postprocessor,
    modeling.embedding_lookup
    """

    def __init__(self,
                 vocab_size,
                 embedding_size=128,
                 use_token_type=False,
                 token_type_vocab_size=16,
                 use_position_embeddings=True,
                 initializer_range=0.02,
                 max_position_embeddings=512,
                 dropout_prob=0.1):
        """
        Args:
            use_token_type: bool. Whether to add embeddings for
                `token_type_ids`.
            token_type_vocab_size: int. The vocabulary size of
                `token_type_ids`.
            use_position_embeddings: bool. Whether to add position embeddings
                for the position of each token in the sequence.
            initializer_range: float. Range of the weight initialization.
            max_position_embeddings: int. Maximum sequence length that might
                ever be used with this model. This can be longer than the
                sequence length of input_tensor, but cannot be shorter.
            dropout_prob: float. Dropout probability applied to the final
                output tensor.

        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.use_token_type = use_token_type
        self.token_type_vocab_size = token_type_vocab_size
        self.use_position_embeddings = use_position_embeddings
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.dropout_prob = dropout_prob

        # modules
        self.word_embeddings = nn.Embedding(
            self.vocab_size, self.embedding_size)

        if self.use_token_type:
            # tf code use one hot multiplication instead.
            self.token_type_embeddings = nn.Embedding(
                self.token_type_vocab_size, self.embedding_size)

        if self.use_position_embeddings:
            self.position_embeddings = nn.Embedding(
                self.max_position_embeddings, self.embedding_size)

        self.layer_norm = nn.LayerNorm(self.embedding_size)
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        """
        Forward.

        Args:
            input_ids: LongTensor of shape [batch_size, seq_length] containing
                word ids.
            token_type_ids: (optional) LongTensor of shape [batch_size,
                seq_length]. Must be specified if `use_token_type` is True.

        """
        assert len(input_ids.shape) == 2
        batch_size, seq_length = input_ids.shape

        if seq_length > self.max_position_embeddings:
            raise ValueError("The seq length (%d) cannot be greater than "
                             "`max_position_embeddings` (%d)" %
                             (seq_length, self.max_position_embeddings))

        output = self.word_embeddings(input_ids)
        if self.use_token_type:
            if token_type_ids is None:
                raise ValueError("`token_type_ids` must be specified if"
                                 "`use_token_type` is True.")
            output += self.token_type_embeddings(token_type_ids)

        if self.use_position_embeddings:
            # hack from the original code:
            # So self.position_embeddings.weight is effectively embedding table
            # for position [0, 1, ..., max_position_embeddings-1], and current
            # sequence has positions [0, 1, 2, ... seq_length-1], so we can
            # just perform a slice.

            position_embeddings = self.position_embeddings.weight[:seq_length]

            # unsqueeze(0) is for broadcasting
            output += position_embeddings.unsqueeze(0)

        output = self.layer_norm(output)
        output = self.dropout(output)
        return output
