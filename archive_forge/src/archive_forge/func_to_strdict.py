import inspect
import math
import re
import warnings
from collections import OrderedDict
from copy import deepcopy
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torchvision
from torch import fx, nn
from torch.fx.graph_module import _copy_attr
def to_strdict(n) -> Dict[str, str]:
    if isinstance(n, list):
        return {str(i): str(i) for i in n}
    return {str(k): str(v) for k, v in n.items()}