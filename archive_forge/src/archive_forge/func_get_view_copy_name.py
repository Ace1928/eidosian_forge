import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
def get_view_copy_name(f: NativeFunction) -> 'OperatorName':
    list_of_ops_with_explicit_view_copy_operators = ['narrow']
    if str(f.func.name) not in list_of_ops_with_explicit_view_copy_operators:
        assert gets_generated_view_copy(f)
    base_name = f'{f.func.name.name.base}_copy'
    view_copy_name = OperatorName(name=BaseOperatorName(base=base_name, inplace=False, dunder_method=f.func.name.name.dunder_method), overload_name=f.func.name.overload_name)
    return view_copy_name