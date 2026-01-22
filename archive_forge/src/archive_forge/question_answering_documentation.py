import inspect
import types
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import numpy as np
from ..data import SquadExample, SquadFeatures, squad_convert_examples_to_features
from ..modelcard import ModelCard
from ..tokenization_utils import PreTrainedTokenizer
from ..utils import (
from .base import ArgumentHandler, ChunkPipeline, build_pipeline_init_args

        When decoding from token probabilities, this method maps token indexes to actual word in the initial context.

        Args:
            text (`str`): The actual context to extract the answer from.
            start (`int`): The answer starting token index.
            end (`int`): The answer end token index.

        Returns:
            Dictionary like `{'answer': str, 'start': int, 'end': int}`
        