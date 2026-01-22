from pathlib import Path
from typing import Union
import numpy as np
from collections import OrderedDict
from minerl.herobraine.hero import spaces
from minerl.herobraine.wrappers.vector_wrapper import Vectorized
from minerl.herobraine.wrapper import EnvWrapper
import copy
import os
Gets the obfuscator from a directory.

        Args:
            obfuscator_dir (Union[str, Path]): The directory containg the pickled obfuscators.
        