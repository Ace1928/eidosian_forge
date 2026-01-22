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
def resume_training(self, *, sgd: Optional[Optimizer]=None) -> Optimizer:
    """Continue training a pretrained model.

        Create and return an optimizer, and initialize "rehearsal" for any pipeline
        component that has a .rehearse() method. Rehearsal is used to prevent
        models from "forgetting" their initialized "knowledge". To perform
        rehearsal, collect samples of text you want the models to retain performance
        on, and call nlp.rehearse() with a batch of Example objects.

        RETURNS (Optimizer): The optimizer.

        DOCS: https://spacy.io/api/language#resume_training
        """
    ops = get_current_ops()
    if self.vocab.vectors.shape[1] >= 1:
        self.vocab.vectors.to_ops(ops)
    for name, proc in self.pipeline:
        if hasattr(proc, '_rehearsal_model'):
            proc._rehearsal_model = deepcopy(proc.model)
    if sgd is not None:
        self._optimizer = sgd
    elif self._optimizer is None:
        self._optimizer = self.create_optimizer()
    return self._optimizer