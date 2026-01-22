from typing import List, Optional, Tuple
import torch
from torchaudio.models import Wav2Vec2Model
from torchaudio.models.emformer import Emformer
from torchaudio.models.rnnt import _TimeReduction
def emformer_hubert_base(extractor_input_dim: int=80, extractor_output_dim: int=128, encoder_dropout: float=0.1, aux_num_out: Optional[int]=None) -> Wav2Vec2Model:
    """Build Emformer HuBERT Model with 20 Emformer layers.

    Args:
        extractor_input_dim (int, optional): The input dimension for feature extractor. (Default: 80)
        extractor_output_dim (int, optional): The output dimension after feature extractor. (Default: 128)
        encoder_dropout (float, optional): Dropout probability in Emformer. (Default: 0.1)
        aux_num_out (int or None, optional): Output dimension of aux layer for fine-tuning. (Default: ``None``)

    Returns:
        Wav2Vec2Model:
            The resulting :py:class:`torchaudio.models.Wav2Vec2Model` model
            with a :py:class:`torchaudio.models.Emformer` encoder.
    """
    return emformer_hubert_model(extractor_input_dim=extractor_input_dim, extractor_output_dim=extractor_output_dim, extractor_use_bias=False, extractor_stride=4, encoder_input_dim=512, encoder_output_dim=1024, encoder_num_heads=8, encoder_ffn_dim=2048, encoder_num_layers=20, encoder_segment_length=4, encoder_left_context_length=30, encoder_right_context_length=1, encoder_dropout=encoder_dropout, encoder_activation='gelu', encoder_max_memory_size=0, encoder_weight_init_scale_strategy='depthwise', encoder_tanh_on_mem=True, aux_num_out=aux_num_out)