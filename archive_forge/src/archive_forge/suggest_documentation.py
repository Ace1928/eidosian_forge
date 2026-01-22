from typing import Any, List, Optional, Tuple
from fugue import FugueWorkflow
from fugue.exceptions import FugueDataFrameError
from triad import assert_or_throw
from tune._utils import from_base64
from tune.api.factory import TUNE_OBJECT_FACTORY, parse_logger
from tune.api.optimize import (
from tune.concepts.flow import TrialReport
from tune.concepts.logger import make_logger
from tune.concepts.space import Space
from tune.constants import TUNE_DATASET_DF_DEFAULT_NAME, TUNE_REPORT, TUNE_REPORT_METRIC
from tune.exceptions import TuneCompileError
Given non-iterative ``objective``, ``space`` and (optional) dataframe,
    suggest the best parameter combinations.

    .. important::
        Please read
        :ref:`Non-Iterative Tuning Guide </notebooks/noniterative.ipynb>`

    :param objective: |NonIterativeObjective|
    :param space: search space, please read |SpaceTutorial|
    :param df: |DataFrameLike|, defaults to None
    :param df_name: dataframe name, defaults to the value of
      ``TUNE_DATASET_DF_DEFAULT_NAME``
    :param temp_path: |TempPath|, defaults to ""
    :param partition_keys: partition keys for ``df``, defaults to None.
      For details, please read |DatasetTutorial|
    :param top_n: number of best results to return, defaults to 1.
      If `<=0` all results will be returned
    :param local_optimizer: |NonIterativeOptimizer|, defaults to None
    :param logger: |LoggerLikeObject|, defaults to None
    :param monitor: realtime monitor, defaults to None. Read
      :ref:`Monitoring Guide </notebooks/noniterative.ipynb#Realtime-Monitoring>`
    :param stopper: early stopper, defaults to None. Read
      :ref:`Early Stopping Guide </notebooks/noniterative.ipynb#Early-Stopping>`
    :param stop_check_interval: an object that can be converted to timedelta,
      defaults to None.
      For details, read :func:`~triad.utils.convert.to_timedelta`
    :param distributed: whether to use the exeuction engine to run different trials
      distributedly, defaults to None. If None, it's equal to True.
    :param shuffle_candidates: whether to shuffle the candidate configurations,
      defaults to True. This is no effect on final result.
    :param execution_engine: Fugue |ExecutionEngine| like object, defaults to None.
      If None, :class:`~fugue.execution.native_execution_engine.NativeExecutionEngine`
      will be used, the task will be running on local machine.
    :param execution_engine_conf: |ParamsLikeObject|, defaults to None
    :return: a list of best results
    