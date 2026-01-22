from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import torch
from torchaudio.models import Emformer
def emformer_rnnt_model(*, input_dim: int, encoding_dim: int, num_symbols: int, segment_length: int, right_context_length: int, time_reduction_input_dim: int, time_reduction_stride: int, transformer_num_heads: int, transformer_ffn_dim: int, transformer_num_layers: int, transformer_dropout: float, transformer_activation: str, transformer_left_context_length: int, transformer_max_memory_size: int, transformer_weight_init_scale_strategy: str, transformer_tanh_on_mem: bool, symbol_embedding_dim: int, num_lstm_layers: int, lstm_layer_norm: bool, lstm_layer_norm_epsilon: float, lstm_dropout: float) -> RNNT:
    """Builds Emformer-based :class:`~torchaudio.models.RNNT`.

    Note:
        For non-streaming inference, the expectation is for `transcribe` to be called on input
        sequences right-concatenated with `right_context_length` frames.

        For streaming inference, the expectation is for `transcribe_streaming` to be called
        on input chunks comprising `segment_length` frames right-concatenated with `right_context_length`
        frames.

    Args:
        input_dim (int): dimension of input sequence frames passed to transcription network.
        encoding_dim (int): dimension of transcription- and prediction-network-generated encodings
            passed to joint network.
        num_symbols (int): cardinality of set of target tokens.
        segment_length (int): length of input segment expressed as number of frames.
        right_context_length (int): length of right context expressed as number of frames.
        time_reduction_input_dim (int): dimension to scale each element in input sequences to
            prior to applying time reduction block.
        time_reduction_stride (int): factor by which to reduce length of input sequence.
        transformer_num_heads (int): number of attention heads in each Emformer layer.
        transformer_ffn_dim (int): hidden layer dimension of each Emformer layer's feedforward network.
        transformer_num_layers (int): number of Emformer layers to instantiate.
        transformer_left_context_length (int): length of left context considered by Emformer.
        transformer_dropout (float): Emformer dropout probability.
        transformer_activation (str): activation function to use in each Emformer layer's
            feedforward network. Must be one of ("relu", "gelu", "silu").
        transformer_max_memory_size (int): maximum number of memory elements to use.
        transformer_weight_init_scale_strategy (str): per-layer weight initialization scaling
            strategy. Must be one of ("depthwise", "constant", ``None``).
        transformer_tanh_on_mem (bool): if ``True``, applies tanh to memory elements.
        symbol_embedding_dim (int): dimension of each target token embedding.
        num_lstm_layers (int): number of LSTM layers to instantiate.
        lstm_layer_norm (bool): if ``True``, enables layer normalization for LSTM layers.
        lstm_layer_norm_epsilon (float): value of epsilon to use in LSTM layer normalization layers.
        lstm_dropout (float): LSTM dropout probability.

    Returns:
        RNNT:
            Emformer RNN-T model.
    """
    encoder = _EmformerEncoder(input_dim=input_dim, output_dim=encoding_dim, segment_length=segment_length, right_context_length=right_context_length, time_reduction_input_dim=time_reduction_input_dim, time_reduction_stride=time_reduction_stride, transformer_num_heads=transformer_num_heads, transformer_ffn_dim=transformer_ffn_dim, transformer_num_layers=transformer_num_layers, transformer_dropout=transformer_dropout, transformer_activation=transformer_activation, transformer_left_context_length=transformer_left_context_length, transformer_max_memory_size=transformer_max_memory_size, transformer_weight_init_scale_strategy=transformer_weight_init_scale_strategy, transformer_tanh_on_mem=transformer_tanh_on_mem)
    predictor = _Predictor(num_symbols, encoding_dim, symbol_embedding_dim=symbol_embedding_dim, num_lstm_layers=num_lstm_layers, lstm_hidden_dim=symbol_embedding_dim, lstm_layer_norm=lstm_layer_norm, lstm_layer_norm_epsilon=lstm_layer_norm_epsilon, lstm_dropout=lstm_dropout)
    joiner = _Joiner(encoding_dim, num_symbols)
    return RNNT(encoder, predictor, joiner)