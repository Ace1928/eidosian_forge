import copy
import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from packaging import version
from . import __version__
from .dynamic_module_utils import custom_object_save
from .utils import (
@num_labels.setter
def num_labels(self, num_labels: int):
    if not hasattr(self, 'id2label') or self.id2label is None or len(self.id2label) != num_labels:
        self.id2label = {i: f'LABEL_{i}' for i in range(num_labels)}
        self.label2id = dict(zip(self.id2label.values(), self.id2label.keys()))