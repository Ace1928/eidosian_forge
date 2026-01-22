from typing import Any, Callable, List, Optional, Type, Union
import pandas as pd
from triad import ParamDict, assert_or_throw
from triad.utils.convert import to_instance
from .._utils.registry import fugue_plugin
from ..exceptions import FuguePluginsRegistrationError
from .execution_engine import (
from .native_execution_engine import NativeExecutionEngine
@fugue_plugin
def infer_execution_engine(obj: List[Any]) -> Any:
    """Infer the correspondent ExecutionEngine based on the input objects. This is
    used in express functions.

    :param objs: the objects
    :return: if the inference succeeded, it returns an object that can be used by
      :func:`~.parse_execution_engine` in the ``engine`` field to construct an
      ExecutionEngine. Otherwise, it returns None.

    .. admonition:: Examples

        .. code-block:: python

            from fugue import transform

            transform(spark_df)

        In this example, the SparkExecutionEngine is inferred from spark_df, it
        is equivalent to:

        .. code-block:: python

            from fugue import transform

            transform(spark_df, engine=current_spark_session)
    """
    return None