import difflib
import json
import os
import re
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import date
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple, Union
import yaml
from ..models import auto as auto_module
from ..models.auto.configuration_auto import model_type_to_module_name
from ..utils import is_flax_available, is_tf_available, is_torch_available, logging
from . import BaseTransformersCLICommand
def replace_model_patterns(text: str, old_model_patterns: ModelPatterns, new_model_patterns: ModelPatterns) -> Tuple[str, str]:
    """
    Replace all patterns present in a given text.

    Args:
        text (`str`): The text to treat.
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
        new_model_patterns (`ModelPatterns`): The patterns for the new model.

    Returns:
        `Tuple(str, str)`: A tuple of with the treated text and the replacement actually done in it.
    """
    attributes_to_check = ['config_class']
    for attr in ['tokenizer_class', 'image_processor_class', 'feature_extractor_class', 'processor_class']:
        if getattr(old_model_patterns, attr) is not None and getattr(new_model_patterns, attr) is not None:
            attributes_to_check.append(attr)
    if old_model_patterns.checkpoint not in [old_model_patterns.model_type, old_model_patterns.model_lower_cased]:
        attributes_to_check.append('checkpoint')
    if old_model_patterns.model_type != old_model_patterns.model_lower_cased:
        attributes_to_check.append('model_type')
    else:
        text = re.sub(f'(\\s*)model_type = "{old_model_patterns.model_type}"', '\\1model_type = "[MODEL_TYPE]"', text)
    if old_model_patterns.model_upper_cased == old_model_patterns.model_camel_cased:
        old_model_value = old_model_patterns.model_upper_cased
        if re.search(f'{old_model_value}_[A-Z_]*[^A-Z_]', text) is not None:
            text = re.sub(f'{old_model_value}([A-Z_]*)([^a-zA-Z_])', '[MODEL_UPPER_CASED]\\1\\2', text)
    else:
        attributes_to_check.append('model_upper_cased')
    attributes_to_check.extend(['model_camel_cased', 'model_lower_cased', 'model_name'])
    for attr in attributes_to_check:
        text = text.replace(getattr(old_model_patterns, attr), ATTRIBUTE_TO_PLACEHOLDER[attr])
    replacements = []
    for attr, placeholder in ATTRIBUTE_TO_PLACEHOLDER.items():
        if placeholder in text:
            replacements.append((getattr(old_model_patterns, attr), getattr(new_model_patterns, attr)))
            text = text.replace(placeholder, getattr(new_model_patterns, attr))
    old_replacement_values = [old for old, new in replacements]
    if len(set(old_replacement_values)) != len(old_replacement_values):
        return (text, '')
    replacements = simplify_replacements(replacements)
    replacements = [f'{old}->{new}' for old, new in replacements]
    return (text, ','.join(replacements))