from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import yaml
from torchgen.model import NativeFunction
from torchgen.selective_build.operator import (
@staticmethod
def get_nop_selector() -> 'SelectiveBuilder':
    return SelectiveBuilder.from_yaml_dict({'include_all_operators': True})