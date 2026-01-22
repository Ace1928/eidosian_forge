from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import yaml
from torchgen.model import NativeFunction
from torchgen.selective_build.operator import (
def is_root_operator(self, name: str) -> bool:
    if not self.is_operator_selected(name):
        return False
    if self.include_all_operators:
        return True
    if name in self.operators:
        op: SelectiveBuildOperator = self.operators[name]
        return op.is_root_operator
    name = strip_operator_overload_name(name)
    if name not in self.operators:
        return False
    base_op: SelectiveBuildOperator = self.operators[name]
    return base_op.include_all_overloads and base_op.is_root_operator