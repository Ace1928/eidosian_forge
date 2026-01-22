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
def duplicate_doc_file(doc_file: Union[str, os.PathLike], old_model_patterns: ModelPatterns, new_model_patterns: ModelPatterns, dest_file: Optional[Union[str, os.PathLike]]=None, frameworks: Optional[List[str]]=None):
    """
    Duplicate a documentation file and adapts it for a new model.

    Args:
        module_file (`str` or `os.PathLike`): Path to the doc file to duplicate.
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
        dest_file (`str` or `os.PathLike`, *optional*): Path to the new doc file.
            Will default to the a file named `{new_model_patterns.model_type}.md` in the same folder as `module_file`.
        frameworks (`List[str]`, *optional*):
            If passed, will only keep the model classes corresponding to this list of frameworks in the new doc file.
    """
    with open(doc_file, 'r', encoding='utf-8') as f:
        content = f.read()
    content = re.sub('<!--\\s*Copyright (\\d+)\\s', f'<!--Copyright {CURRENT_YEAR} ', content)
    if frameworks is None:
        frameworks = get_default_frameworks()
    if dest_file is None:
        dest_file = Path(doc_file).parent / f'{new_model_patterns.model_type}.md'
    lines = content.split('\n')
    blocks = []
    current_block = []
    for line in lines:
        if line.startswith('#'):
            blocks.append('\n'.join(current_block))
            current_block = [line]
        else:
            current_block.append(line)
    blocks.append('\n'.join(current_block))
    new_blocks = []
    in_classes = False
    for block in blocks:
        if not block.startswith('#'):
            new_blocks.append(block)
        elif re.search('^#\\s+\\S+', block) is not None:
            new_blocks.append(f'# {new_model_patterns.model_name}\n')
        elif not in_classes and old_model_patterns.config_class in block.split('\n')[0]:
            in_classes = True
            new_blocks.append(DOC_OVERVIEW_TEMPLATE.format(model_name=new_model_patterns.model_name))
            new_block, _ = replace_model_patterns(block, old_model_patterns, new_model_patterns)
            new_blocks.append(new_block)
        elif in_classes:
            in_classes = True
            block_title = block.split('\n')[0]
            block_class = re.search('^#+\\s+(\\S.*)$', block_title).groups()[0]
            new_block, _ = replace_model_patterns(block, old_model_patterns, new_model_patterns)
            if 'Tokenizer' in block_class:
                if old_model_patterns.tokenizer_class != new_model_patterns.tokenizer_class:
                    new_blocks.append(new_block)
            elif 'ImageProcessor' in block_class:
                if old_model_patterns.image_processor_class != new_model_patterns.image_processor_class:
                    new_blocks.append(new_block)
            elif 'FeatureExtractor' in block_class:
                if old_model_patterns.feature_extractor_class != new_model_patterns.feature_extractor_class:
                    new_blocks.append(new_block)
            elif 'Processor' in block_class:
                if old_model_patterns.processor_class != new_model_patterns.processor_class:
                    new_blocks.append(new_block)
            elif block_class.startswith('Flax'):
                if 'flax' in frameworks:
                    new_blocks.append(new_block)
            elif block_class.startswith('TF'):
                if 'tf' in frameworks:
                    new_blocks.append(new_block)
            elif len(block_class.split(' ')) == 1:
                if 'pt' in frameworks:
                    new_blocks.append(new_block)
            else:
                new_blocks.append(new_block)
    with open(dest_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_blocks))