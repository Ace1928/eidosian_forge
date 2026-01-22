import itertools
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torchgen.api.dispatcher as dispatcher
from torchgen.api.lazy import (
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import method_with_native_function
from torchgen.dest.lazy_ts_lowering import ts_lowering_body
from torchgen.model import (
def aten_symbol(schema: LazyIrSchema) -> str:
    missing_interned_strings = {'sigmoid_backward'}
    if schema.aten_name in missing_interned_strings:
        return f'c10::Symbol::fromQualString("aten::{schema.aten_name}")'
    if not schema.aten_name.startswith('at::'):
        return f'at::aten::{schema.aten_name}'
    else:
        return schema.aten_name