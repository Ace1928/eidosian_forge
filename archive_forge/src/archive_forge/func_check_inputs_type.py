import inspect
import re
import string
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
def check_inputs_type(x):
    if not isinstance(x, (ExportArgs, tuple)):
        raise ValueError(f'Expecting inputs type to be either a tuple, or ExportArgs, got: {type(x)}')