import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
def native_function_class_name(self) -> Optional[str]:
    if self.external:
        return f'{str(self.dispatch_key)}NativeFunctions'
    else:
        return None