import math
from typing import Dict, List, Optional, Tuple
import torch
from torchaudio.models import Conformer, RNNT
from torchaudio.models.rnnt import _Joiner, _Predictor, _TimeReduction, _Transcriber
def conformer_rnnt_biasing_base(charlist=None, biasing=True) -> RNNT:
    """Builds basic version of Conformer RNN-T model with TCPGen.

    Returns:
        RNNT:
            Conformer RNN-T model with TCPGen-based biasing support.
    """
    return conformer_rnnt_biasing(input_dim=80, encoding_dim=576, time_reduction_stride=4, conformer_input_dim=144, conformer_ffn_dim=576, conformer_num_layers=16, conformer_num_heads=4, conformer_depthwise_conv_kernel_size=31, conformer_dropout=0.1, num_symbols=601, symbol_embedding_dim=256, num_lstm_layers=1, lstm_hidden_dim=320, lstm_layer_norm=True, lstm_layer_norm_epsilon=1e-05, lstm_dropout=0.3, joiner_activation='tanh', attndim=256, biasing=biasing, charlist=charlist, deepbiasing=True, tcpsche=30, DBaverage=False)