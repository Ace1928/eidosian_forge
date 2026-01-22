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
def remove_pipe(self, name: str) -> Tuple[str, PipeCallable]:
    """Remove a component from the pipeline.

        name (str): Name of the component to remove.
        RETURNS (Tuple[str, Callable[[Doc], Doc]]): A `(name, component)` tuple of the removed component.

        DOCS: https://spacy.io/api/language#remove_pipe
        """
    if name not in self.component_names:
        raise ValueError(Errors.E001.format(name=name, opts=self.component_names))
    removed = self._components.pop(self.component_names.index(name))
    self._pipe_meta.pop(name)
    self._pipe_configs.pop(name)
    self.meta.get('_sourced_vectors_hashes', {}).pop(name, None)
    if name in self._config['initialize']['components']:
        self._config['initialize']['components'].pop(name)
    if name in self.disabled:
        self._disabled.remove(name)
    self._link_components()
    return removed