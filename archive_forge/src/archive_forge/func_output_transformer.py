import copy
from typing import Any, Callable, Dict, List, Optional, Type, Union, no_type_check
from triad import ParamDict, Schema
from triad.utils.assertion import assert_arg_not_none, assert_or_throw
from triad.utils.convert import get_caller_global_local_vars, to_function, to_instance
from triad.utils.hash import to_uuid
from fugue._utils.interfaceless import is_class_method, parse_output_schema_from_comment
from fugue._utils.registry import fugue_plugin
from fugue.dataframe import ArrayDataFrame, DataFrame, DataFrames, LocalDataFrame
from fugue.dataframe.function_wrapper import DataFrameFunctionWrapper
from fugue.exceptions import FugueInterfacelessError
from fugue.extensions.transformer.constants import OUTPUT_TRANSFORMER_DUMMY_SCHEMA
from fugue.extensions.transformer.transformer import CoTransformer, Transformer
from .._utils import (
def output_transformer(**validation_rules: Any) -> Callable[[Any], '_FuncAsTransformer']:
    """Decorator for transformers

    Please read |TransformerTutorial|
    """

    def deco(func: Callable) -> '_FuncAsOutputTransformer':
        assert_or_throw(not is_class_method(func), NotImplementedError("output_transformer decorator can't be used on class methods"))
        return _FuncAsOutputTransformer.from_func(func, schema=None, validation_rules=to_validation_rules(validation_rules))
    return deco