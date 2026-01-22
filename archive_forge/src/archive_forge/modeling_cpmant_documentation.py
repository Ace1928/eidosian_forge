import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_cpmant import CpmAntConfig

        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, seq_len)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`CPMAntTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                CPMAnt will process attention mask automatically, this parameter is a dummy parameter for
                text-generation pipeline.

        Example:

        Text Generation with CpmAntForCausalLM.
        ```python
        >>> from transformers import CPMAntTokenizer, CpmAntForCausalLM

        >>> texts = "今天天气不错，"
        >>> model = CpmAntForCausalLM.from_pretrained("openbmb/cpm-ant-10b")
        >>> tokenizer = CPMAntTokenizer.from_pretrained("openbmb/cpm-ant-10b")
        >>> input_ids = tokenizer(texts, return_tensors="pt")
        >>> outputs = model.generate(**input_ids)
        >>> output_texts = tokenizer.batch_decode(outputs)
        >>> print(output_texts)
        ['今天天气不错，阳光明媚，我和妈妈一起去超市买东西。\n在超市里，我看到了一个很好玩的玩具，它的名字叫“机器人”。它有一个圆圆的脑袋，两只圆圆的眼睛，还有一个圆圆的']
        ```
        