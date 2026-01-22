from typing import Any, Dict, List, Optional
from triad.utils.assertion import assert_or_throw
from ..collections.yielded import Yielded
from ..constants import FUGUE_CONF_WORKFLOW_EXCEPTION_INJECT
from ..dataframe import DataFrame, AnyDataFrame
from ..dataframe.api import get_native_as_df
from ..exceptions import FugueInterfacelessError, FugueWorkflowCompileError
from ..execution import make_execution_engine
from .workflow import FugueWorkflow
def out_transform(df: Any, using: Any, params: Any=None, partition: Any=None, callback: Any=None, ignore_errors: Optional[List[Any]]=None, engine: Any=None, engine_conf: Any=None) -> None:
    """Transform this dataframe using transformer. It's a wrapper of
    :meth:`~fugue.workflow.workflow.FugueWorkflow.out_transform` and
    :meth:`~fugue.workflow.workflow.FugueWorkflow.run`. It will let you do the
    basic dataframe transformation without using
    :class:`~fugue.workflow.workflow.FugueWorkflow` and
    :class:`~fugue.dataframe.dataframe.DataFrame`. Only native types are
    accepted for both input and output.

    Please read |TransformerTutorial|

    :param df: |DataFrameLikeObject| or :class:`~fugue.workflow.yielded.Yielded`
        or a path string to a parquet file
    :param using: transformer-like object, can't be a string expression
    :param params: |ParamsLikeObject| to run the processor, defaults to None
        The transformer will be able to access this value from
        :meth:`~fugue.extensions.context.ExtensionContext.params`
    :param partition: |PartitionLikeObject|, defaults to None
    :param callback: |RPCHandlerLikeObject|, defaults to None
    :param ignore_errors: list of exception types the transformer can ignore,
        defaults to None (empty list)
    :param engine: it can be empty string or null (use the default execution
        engine), a string (use the registered execution engine), an
        :class:`~fugue.execution.execution_engine.ExecutionEngine` type, or
        the :class:`~fugue.execution.execution_engine.ExecutionEngine` instance
        , or a tuple of two values where the first value represents execution
        engine and the second value represents the sql engine (you can use ``None``
        for either of them to use the default one), defaults to None
    :param engine_conf: |ParamsLikeObject|, defaults to None

    .. note::

        This function can only take parquet file paths in `df`. CSV and JSON file
        formats are disallowed.

        This transformation is guaranteed to execute immediately (eager)
        and return nothing
    """
    dag = FugueWorkflow(compile_conf={FUGUE_CONF_WORKFLOW_EXCEPTION_INJECT: 0})
    try:
        src = dag.create(df)
    except FugueWorkflowCompileError:
        if isinstance(df, str):
            src = dag.load(df, fmt='parquet')
        else:
            raise
    src.out_transform(using=using, params=params, pre_partition=partition, callback=callback, ignore_errors=ignore_errors or [])
    dag.run(make_execution_engine(engine, conf=engine_conf, infer_by=[df]))