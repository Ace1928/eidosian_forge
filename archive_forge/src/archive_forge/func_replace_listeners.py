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
def replace_listeners(self, tok2vec_name: str, pipe_name: str, listeners: Iterable[str]) -> None:
    """Find listener layers (connecting to a token-to-vector embedding
        component) of a given pipeline component model and replace
        them with a standalone copy of the token-to-vector layer. This can be
        useful when training a pipeline with components sourced from an existing
        pipeline: if multiple components (e.g. tagger, parser, NER) listen to
        the same tok2vec component, but some of them are frozen and not updated,
        their performance may degrade significantly as the tok2vec component is
        updated with new data. To prevent this, listeners can be replaced with
        a standalone tok2vec layer that is owned by the component and doesn't
        change if the component isn't updated.

        tok2vec_name (str): Name of the token-to-vector component, typically
            "tok2vec" or "transformer".
        pipe_name (str): Name of pipeline component to replace listeners for.
        listeners (Iterable[str]): The paths to the listeners, relative to the
            component config, e.g. ["model.tok2vec"]. Typically, implementations
            will only connect to one tok2vec component, [model.tok2vec], but in
            theory, custom models can use multiple listeners. The value here can
            either be an empty list to not replace any listeners, or a complete
            (!) list of the paths to all listener layers used by the model.

        DOCS: https://spacy.io/api/language#replace_listeners
        """
    if tok2vec_name not in self.pipe_names:
        err = Errors.E889.format(tok2vec=tok2vec_name, name=pipe_name, unknown=tok2vec_name, opts=', '.join(self.pipe_names))
        raise ValueError(err)
    if pipe_name not in self.pipe_names:
        err = Errors.E889.format(tok2vec=tok2vec_name, name=pipe_name, unknown=pipe_name, opts=', '.join(self.pipe_names))
        raise ValueError(err)
    tok2vec = self.get_pipe(tok2vec_name)
    tok2vec_cfg = self.get_pipe_config(tok2vec_name)
    if not isinstance(tok2vec, ty.ListenedToComponent):
        raise ValueError(Errors.E888.format(name=tok2vec_name, pipe=type(tok2vec)))
    tok2vec_model = tok2vec.model
    pipe_listeners = tok2vec.listener_map.get(pipe_name, [])
    pipe = self.get_pipe(pipe_name)
    pipe_cfg = self._pipe_configs[pipe_name]
    if listeners:
        util.logger.debug("Replacing listeners of component '%s'", pipe_name)
        if len(list(listeners)) != len(pipe_listeners):
            err = Errors.E887.format(name=pipe_name, tok2vec=tok2vec_name, paths=listeners, n_listeners=len(pipe_listeners))
            raise ValueError(err)
        for listener_path in listeners:
            try:
                util.dot_to_object(pipe_cfg, listener_path)
            except KeyError:
                err = Errors.E886.format(name=pipe_name, tok2vec=tok2vec_name, path=listener_path)
                raise ValueError(err)
            new_config = tok2vec_cfg['model']
            if 'replace_listener_cfg' in tok2vec_model.attrs:
                replace_func = tok2vec_model.attrs['replace_listener_cfg']
                new_config = replace_func(tok2vec_cfg['model'], pipe_cfg['model']['tok2vec'])
            util.set_dot_to_object(pipe_cfg, listener_path, new_config)
        for listener in pipe_listeners:
            new_model = tok2vec_model.copy()
            replace_listener_func = tok2vec_model.attrs.get('replace_listener')
            if replace_listener_func is not None:
                num_params = len(inspect.signature(replace_listener_func).parameters)
                if num_params == 1:
                    new_model = replace_listener_func(new_model)
                elif num_params == 3:
                    new_model = replace_listener_func(new_model, listener, tok2vec)
                else:
                    raise ValueError(Errors.E1055.format(num_params=num_params))
            util.replace_model_node(pipe.model, listener, new_model)
            tok2vec.remove_listener(listener, pipe_name)