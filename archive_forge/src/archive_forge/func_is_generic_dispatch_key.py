import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
def is_generic_dispatch_key(dk: DispatchKey) -> bool:
    return dk in {DispatchKey.CompositeExplicitAutograd, DispatchKey.CompositeExplicitAutogradNonFunctional, DispatchKey.CompositeImplicitAutograd, DispatchKey.CompositeImplicitAutogradNestedTensor}