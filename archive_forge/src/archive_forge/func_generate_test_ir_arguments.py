import json
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.cpp as cpp
from torchgen.context import native_function_manager
from torchgen.model import (
from torchgen.static_runtime import config
def generate_test_ir_arguments(schema: FunctionSchema) -> List[Tuple[str, Optional[str]]]:

    def ir_argument(arg: Argument) -> Tuple[str, Optional[str]]:
        t = arg.type
        add_optional = False
        if isinstance(t, OptionalType):
            t = t.elem
            add_optional = True
        assert isinstance(t, BaseType)
        type_str = None
        if t.name in generate_test_ir_arguments_base_ty_to_type_str_:
            type_str = generate_test_ir_arguments_base_ty_to_type_str_[t.name]
        if type_str and add_optional:
            type_str = f'{type_str}?'
        return ('%' + arg.name, type_str)
    return [ir_argument(arg) for arg in schema.schema_order_arguments()]