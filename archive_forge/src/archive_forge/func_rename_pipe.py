import functools
import inspect
import itertools
import multiprocessing as mp
import random
import traceback
import warnings
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain, cycle
from pathlib import Path
from timeit import default_timer as timer
from typing import (
import srsly
from thinc.api import Config, CupyOps, Optimizer, get_current_ops
from . import about, ty, util
from .compat import Literal
from .errors import Errors, Warnings
from .git_info import GIT_VERSION
from .lang.punctuation import TOKENIZER_INFIXES, TOKENIZER_PREFIXES, TOKENIZER_SUFFIXES
from .lang.tokenizer_exceptions import BASE_EXCEPTIONS, URL_MATCH
from .lookups import load_lookups
from .pipe_analysis import analyze_pipes, print_pipe_analysis, validate_attrs
from .schemas import (
from .scorer import Scorer
from .tokenizer import Tokenizer
from .tokens import Doc
from .tokens.underscore import Underscore
from .training import Example, validate_examples
from .training.initialize import init_tok2vec, init_vocab
from .util import (
from .vectors import BaseVectors
from .vocab import Vocab, create_vocab
def rename_pipe(self, old_name: str, new_name: str) -> None:
    """Rename a pipeline component.

        old_name (str): Name of the component to rename.
        new_name (str): New name of the component.

        DOCS: https://spacy.io/api/language#rename_pipe
        """
    if old_name not in self.component_names:
        raise ValueError(Errors.E001.format(name=old_name, opts=self.component_names))
    if new_name in self.component_names:
        raise ValueError(Errors.E007.format(name=new_name, opts=self.component_names))
    i = self.component_names.index(old_name)
    self._components[i] = (new_name, self._components[i][1])
    self._pipe_meta[new_name] = self._pipe_meta.pop(old_name)
    self._pipe_configs[new_name] = self._pipe_configs.pop(old_name)
    if old_name in self._config['initialize']['components']:
        init_cfg = self._config['initialize']['components'].pop(old_name)
        self._config['initialize']['components'][new_name] = init_cfg
    self._link_components()