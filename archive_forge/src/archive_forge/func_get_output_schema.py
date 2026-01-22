import os
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4
import cloudpickle
import numpy as np
import pandas as pd
from fugue import (
from triad import FileSystem, ParamDict, assert_or_throw, to_uuid
from tune._utils import from_base64, to_base64
from tune.concepts.flow import Trial
from tune.concepts.space import Space
from tune.constants import (
from tune.exceptions import TuneCompileError
def get_output_schema(self, df: DataFrame) -> Any:
    dfn = self.params.get_or_throw('name', str)
    return self.key_schema + f'{TUNE_DATASET_DF_PREFIX}{dfn}:str'