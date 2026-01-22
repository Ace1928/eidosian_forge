import copy
import json
import sys
import warnings
from collections import defaultdict, namedtuple
from dataclasses import (MISSING,
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import (Any, Collection, Mapping, Union, get_type_hints,
from uuid import UUID
from typing_inspect import is_union_type  # type: ignore
from dataclasses_json import cfg
from dataclasses_json.utils import (_get_type_cons, _get_type_origin,
def handle_pep0673(pre_0673_hint: str) -> Union[Type, str]:
    for module in sys.modules:
        maybe_resolved = getattr(sys.modules[module], type_args, None)
        if maybe_resolved:
            return maybe_resolved
    warnings.warn(f'Could not resolve self-reference for type {pre_0673_hint}, decoded type might be incorrect or decode might fail altogether.')
    return pre_0673_hint