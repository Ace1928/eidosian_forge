import __main__
import abc
import glob
import itertools
import keyword
import logging
import os
import re
import rlcompleter
import builtins
from enum import Enum
from typing import (
from . import inspection
from . import line as lineparts
from .line import LinePart
from .lazyre import LazyReCompile
from .simpleeval import safe_eval, evaluate_current_expression, EvaluationError
from .importcompletion import ModuleGatherer
def get_default_completer(mode: AutocompleteModes, module_gatherer: ModuleGatherer) -> Tuple[BaseCompletionType, ...]:
    return (DictKeyCompletion(mode=mode), ImportCompletion(module_gatherer, mode=mode), FilenameCompletion(mode=mode), MagicMethodCompletion(mode=mode), MultilineJediCompletion(mode=mode), CumulativeCompleter((GlobalCompletion(mode=mode), ParameterNameCompletion(mode=mode)), mode=mode), AttrCompletion(mode=mode), ExpressionAttributeCompletion(mode=mode)) if mode != AutocompleteModes.NONE else tuple()