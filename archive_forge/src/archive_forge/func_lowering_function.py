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
def lowering_function(self, schema: LazyIrSchema) -> str:
    signature = '\n  torch::lazy::TSOpVector Lower(\n      std::shared_ptr<torch::jit::GraphFunction> function,\n      torch::lazy::TSLoweringContext* loctx) const override'
    if schema.properties.LowerDeclOnly:
        return f'{signature};'
    elif schema.properties.Lower:
        return f'{signature} {{\n    {ts_lowering_body(schema)}\n  }}\n            '
    else:
        return ''