from typing import Any, Callable, List, Optional, Type, Union
import pandas as pd
from triad import ParamDict, assert_or_throw
from triad.utils.convert import to_instance
from .._utils.registry import fugue_plugin
from ..exceptions import FuguePluginsRegistrationError
from .execution_engine import (
from .native_execution_engine import NativeExecutionEngine
def register_default_sql_engine(func: Callable, on_dup='overwrite') -> None:
    """Register :class:`~fugue.execution.execution_engine.SQLEngine` as the
    default engine

    :param func: a callable taking
      :class:`~fugue.execution.execution_engine.ExecutionEngine`
      and ``**kwargs`` and returning a
      :class:`~fugue.execution.execution_engine.SQLEngine` instance
    :param on_dup: action on duplicated ``name``. It can be "overwrite", "ignore"
      (not overwriting) or "throw" (throw exception), defaults to "overwrite".

    :raises KeyError: if ``on_dup`` is ``throw`` and the ``name`` already exists

    .. note::

        You should be careful to use this function, because when you set a custom
        SQL engine as default, all execution engines you create will use this SQL
        engine unless you are explicit. For example if you set the default SQL engine
        to be a Spark specific one, then if you start a NativeExecutionEngine, it will
        try to use it and will throw exceptions.

        So it's always a better idea to use ``register_sql_engine`` instead

    .. admonition:: Examples

        .. code-block:: python

            # create a new engine with name my (overwrites if existed)
            register_default_sql_engine(lambda engine: MySQLEngine(engine))

            # create NativeExecutionEngine with MySQLEngine as the default
            make_execution_engine()

            # create SparkExecutionEngine with MySQLEngine instead of SparkSQLEngine
            make_execution_engine("spark")

            # NativeExecutionEngine with MySQLEngine
            with FugueWorkflow() as dag:
                dag.create([[0]],"a:int").show()
            dag.run()
    """
    parse_sql_engine.register(func=lambda engine, execution_engine, **kwargs: func(execution_engine, **kwargs), matcher=lambda engine, execution_engine, **kwargs: engine is None or (isinstance(engine, str) and engine == ''), priority=_get_priority(on_dup))