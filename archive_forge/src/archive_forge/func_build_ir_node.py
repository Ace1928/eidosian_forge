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
def build_ir_node(self, func: NativeFunction, schema: LazyIrSchema) -> str:
    node_ctor_input_str = node_ctor_inputs(schema)
    return f'torch::lazy::NodePtr node = torch::lazy::ReuseNode<{schema.node_name}>({node_ctor_input_str});\n        if (!node) {{\n            {self.shape_inference(func, schema)}\n            node = torch::lazy::MakeNode<{schema.node_name}>({node_ctor_input_str}, std::move(shapes));\n            CacheNode(node);\n        }}\n        '