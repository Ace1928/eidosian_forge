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
Remove an existing output directory. Typically used to ensure that that
    a directory like model-best and its contents aren't just being overwritten
    by nlp.to_disk, which could preserve existing subdirectories (e.g.
    components that don't exist anymore).
    