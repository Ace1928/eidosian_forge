import os
import warnings
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from multiprocessing import Pool, get_context, get_start_method
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Union
import numpy as np
from ...processing_utils import ProcessorMixin
from ...utils import ModelOutput, logging, requires_backends
@property
def language_model(self):
    return self.decoder.model_container[self.decoder._model_key]