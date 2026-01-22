import importlib
import json
import os
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union
from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import TOKENIZER_CONFIG_FILE
from ...utils import (
from ..encoder_decoder import EncoderDecoderConfig
from .auto_factory import _LazyAutoMapping
from .configuration_auto import (
def tokenizer_class_from_name(class_name: str):
    if class_name == 'PreTrainedTokenizerFast':
        return PreTrainedTokenizerFast
    for module_name, tokenizers in TOKENIZER_MAPPING_NAMES.items():
        if class_name in tokenizers:
            module_name = model_type_to_module_name(module_name)
            module = importlib.import_module(f'.{module_name}', 'transformers.models')
            try:
                return getattr(module, class_name)
            except AttributeError:
                continue
    for config, tokenizers in TOKENIZER_MAPPING._extra_content.items():
        for tokenizer in tokenizers:
            if getattr(tokenizer, '__name__', None) == class_name:
                return tokenizer
    main_module = importlib.import_module('transformers')
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)
    return None