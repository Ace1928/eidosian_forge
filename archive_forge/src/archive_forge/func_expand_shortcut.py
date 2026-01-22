from __future__ import annotations
import json
import re
import tempfile
import textwrap
import warnings
from pathlib import Path
from typing import Iterable
import huggingface_hub
import semantic_version as semver
from gradio_client.documentation import document
from huggingface_hub import CommitOperationAdd
from gradio.themes.utils import (
from gradio.themes.utils.readme_content import README_CONTENT
def expand_shortcut(shortcut, mode='color', prefix=None):
    if not isinstance(shortcut, str):
        return shortcut
    if mode == 'color':
        for color in colors.Color.all:
            if color.name == shortcut:
                return color
        raise ValueError(f'Color shortcut {shortcut} not found.')
    elif mode == 'size':
        for size in sizes.Size.all:
            if size.name == f'{prefix}_{shortcut}':
                return size
        raise ValueError(f'Size shortcut {shortcut} not found.')