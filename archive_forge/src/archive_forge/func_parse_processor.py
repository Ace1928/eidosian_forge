import copy
from typing import Any, Callable, Dict, List, Optional, no_type_check
from triad import ParamDict, to_uuid
from triad.collections import Schema
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import get_caller_global_local_vars, to_function, to_instance
from fugue._utils.interfaceless import parse_output_schema_from_comment
from fugue._utils.registry import fugue_plugin
from fugue.dataframe import DataFrame, DataFrames
from fugue.dataframe.function_wrapper import DataFrameFunctionWrapper
from fugue.exceptions import FugueInterfacelessError
from fugue.extensions.processor.processor import Processor
from .._utils import (
@fugue_plugin
def parse_processor(obj: Any) -> Any:
    """Parse an object to another object that can be converted to a Fugue
    :class:`~fugue.extensions.processor.processor.Processor`.

    .. admonition:: Examples

        .. code-block:: python

            from fugue import Processor, parse_processor, FugueWorkflow
            from triad import to_uuid

            class My(Processor):
                def __init__(self, x):
                    self.x = x

                def process(self, dfs):
                    raise NotImplementedError

                def __uuid__(self) -> str:
                    return to_uuid(super().__uuid__(), self.x)

            @parse_processor.candidate(
                lambda x: isinstance(x, str) and x.startswith("-*"))
            def _parse(obj):
                return My(obj)

            dag = FugueWorkflow()
            dag.df([[0]], "a:int").process("-*abc")
            # ==  dag.df([[0]], "a:int").process(My("-*abc"))

            dag.run()
    """
    if isinstance(obj, str) and obj in _PROCESSOR_REGISTRY:
        return _PROCESSOR_REGISTRY[obj]
    return obj