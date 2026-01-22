import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
@property
def non_out(self) -> Sequence[Union[Argument, SelfArgument, TensorOptionsArguments]]:
    ret: List[Union[Argument, SelfArgument, TensorOptionsArguments]] = []
    ret.extend(self.positional)
    ret.extend(self.kwarg_only)
    return ret