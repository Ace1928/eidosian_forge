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
def simplify_replacements(replacements):
    """
    Simplify a list of replacement patterns to make sure there are no needless ones.

    For instance in the sequence "Bert->BertNew, BertConfig->BertNewConfig, bert->bert_new", the replacement
    "BertConfig->BertNewConfig" is implied by "Bert->BertNew" so not needed.

    Args:
        replacements (`List[Tuple[str, str]]`): List of patterns (old, new)

    Returns:
        `List[Tuple[str, str]]`: The list of patterns simplified.
    """
    if len(replacements) <= 1:
        return replacements
    replacements.sort(key=lambda x: len(x[0]))
    idx = 0
    while idx < len(replacements):
        old, new = replacements[idx]
        j = idx + 1
        while j < len(replacements):
            old_2, new_2 = replacements[j]
            if old_2.replace(old, new) == new_2:
                replacements.pop(j)
            else:
                j += 1
        idx += 1
    return replacements