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
def extract_block(content: str, indent_level: int=0) -> str:
    """Return the first block in `content` with the indent level `indent_level`.

    The first line in `content` should be indented at `indent_level` level, otherwise an error will be thrown.

    This method will immediately stop the search when a (non-empty) line with indent level less than `indent_level` is
    encountered.

    Args:
        content (`str`): The content to parse
        indent_level (`int`, *optional*, default to 0): The indent level of the blocks to search for

    Returns:
        `str`: The first block in `content` with the indent level `indent_level`.
    """
    current_object = []
    lines = content.split('\n')
    end_markers = [')', ']', '}', '"""']
    for idx, line in enumerate(lines):
        if idx == 0 and indent_level > 0 and (not is_empty_line(line)) and (find_indent(line) != indent_level):
            raise ValueError(f'When `indent_level > 0`, the first line in `content` should have indent level {indent_level}. Got {find_indent(line)} instead.')
        if find_indent(line) < indent_level and (not is_empty_line(line)):
            break
        is_valid_object = len(current_object) > 0
        if not is_empty_line(line) and (not line.endswith(':')) and (find_indent(line) == indent_level) and is_valid_object:
            if line.lstrip() in end_markers:
                current_object.append(line)
            return '\n'.join(current_object)
        else:
            current_object.append(line)
    if len(current_object) > 0:
        return '\n'.join(current_object)