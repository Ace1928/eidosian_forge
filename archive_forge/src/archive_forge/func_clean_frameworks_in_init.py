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
def clean_frameworks_in_init(init_file: Union[str, os.PathLike], frameworks: Optional[List[str]]=None, keep_processing: bool=True):
    """
    Removes all the import lines that don't belong to a given list of frameworks or concern tokenizers/feature
    extractors/image processors/processors in an init.

    Args:
        init_file (`str` or `os.PathLike`): The path to the init to treat.
        frameworks (`List[str]`, *optional*):
           If passed, this will remove all imports that are subject to a framework not in frameworks
        keep_processing (`bool`, *optional*, defaults to `True`):
            Whether or not to keep the preprocessing (tokenizer, feature extractor, image processor, processor) imports
            in the init.
    """
    if frameworks is None:
        frameworks = get_default_frameworks()
    names = {'pt': 'torch'}
    to_remove = [names.get(f, f) for f in ['pt', 'tf', 'flax'] if f not in frameworks]
    if not keep_processing:
        to_remove.extend(['sentencepiece', 'tokenizers', 'vision'])
    if len(to_remove) == 0:
        return
    remove_pattern = '|'.join(to_remove)
    re_conditional_imports = re.compile(f'^\\s*if not is_({remove_pattern})_available\\(\\):\\s*$')
    re_try = re.compile('\\s*try:')
    re_else = re.compile('\\s*else:')
    re_is_xxx_available = re.compile(f'is_({remove_pattern})_available')
    with open(init_file, 'r', encoding='utf-8') as f:
        content = f.read()
    lines = content.split('\n')
    new_lines = []
    idx = 0
    while idx < len(lines):
        if re_conditional_imports.search(lines[idx]) is not None and re_try.search(lines[idx - 1]) is not None:
            new_lines.pop()
            idx += 1
            while is_empty_line(lines[idx]) or re_else.search(lines[idx]) is None:
                idx += 1
            idx += 1
            indent = find_indent(lines[idx])
            while find_indent(lines[idx]) >= indent or is_empty_line(lines[idx]):
                idx += 1
        elif re_is_xxx_available.search(lines[idx]) is not None:
            line = lines[idx]
            for framework in to_remove:
                line = line.replace(f', is_{framework}_available', '')
                line = line.replace(f'is_{framework}_available, ', '')
                line = line.replace(f'is_{framework}_available,', '')
                line = line.replace(f'is_{framework}_available', '')
            if len(line.strip()) > 0:
                new_lines.append(line)
            idx += 1
        elif keep_processing or (re.search('^\\s*"(tokenization|processing|feature_extraction|image_processing)', lines[idx]) is None and re.search('^\\s*from .(tokenization|processing|feature_extraction|image_processing)', lines[idx]) is None):
            new_lines.append(lines[idx])
            idx += 1
        else:
            idx += 1
    with open(init_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))