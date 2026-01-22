import copy
from typing import Any, Callable, Dict, List, Optional, no_type_check
from triad import ParamDict, to_uuid
from triad.utils.convert import get_caller_global_local_vars, to_function, to_instance
from fugue._utils.registry import fugue_plugin
from fugue.dataframe import DataFrames
from fugue.dataframe.function_wrapper import DataFrameFunctionWrapper
from fugue.exceptions import FugueInterfacelessError
from fugue.extensions._utils import (
from fugue.extensions.outputter.outputter import Outputter
def outputter(**validation_rules: Any) -> Callable[[Any], '_FuncAsOutputter']:
    """Decorator for outputters

    Please read
    :doc:`Outputter Tutorial <tutorial:tutorials/extensions/outputter>`
    """

    def deco(func: Callable) -> '_FuncAsOutputter':
        return _FuncAsOutputter.from_func(func, validation_rules=to_validation_rules(validation_rules))
    return deco