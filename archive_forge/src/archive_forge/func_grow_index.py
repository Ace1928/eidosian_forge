import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
@staticmethod
def grow_index(parent_index: Dict[DispatchKey, Dict['OperatorName', BackendMetadata]], child_index: Dict[DispatchKey, Dict['OperatorName', BackendMetadata]]) -> None:
    for k, v in child_index.items():
        for op_name, metadata in v.items():
            assert op_name not in parent_index[k], f'duplicate operator {op_name} for dispatch key {k}'
            parent_index[k][op_name] = metadata