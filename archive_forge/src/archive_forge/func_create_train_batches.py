import random
import shutil
import sys
from pathlib import Path
from timeit import default_timer as timer
from typing import (
from thinc.api import Config, Optimizer, constant, fix_random_seed, set_gpu_allocator
from wasabi import Printer
from ..errors import Errors
from ..schemas import ConfigSchemaTraining
from ..util import logger, registry, resolve_dot_names
from .example import Example
def create_train_batches(nlp: 'Language', corpus: Callable[['Language'], Iterable[Example]], batcher: Callable[[Iterable[Example]], Iterable[Example]], max_epochs: int):
    epoch = 0
    if max_epochs >= 0:
        examples = list(corpus(nlp))
        if not examples:
            raise ValueError(Errors.E986)
    while max_epochs < 1 or epoch != max_epochs:
        if max_epochs >= 0:
            random.shuffle(examples)
        else:
            examples = corpus(nlp)
        for batch in batcher(examples):
            yield (epoch, batch)
        epoch += 1