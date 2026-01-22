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
@property
def validation_rules(self) -> Dict[str, Any]:
    return self._validation_rules