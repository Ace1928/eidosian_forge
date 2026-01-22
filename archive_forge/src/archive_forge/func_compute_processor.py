from copy import copy
from typing import Any, Callable, Dict, Iterable, Optional
from fugue import ArrayDataFrame, DataFrame, ExecutionEngine
from triad import assert_or_throw
from tune._utils import run_monitored_process
from tune.concepts.dataset import StudyResult, TuneDataset, _get_trials_from_row
from tune.concepts.flow import RemoteTrialJudge, TrialCallback, TrialJudge, TrialReport
from tune.concepts.flow.judge import Monitor, NoOpTrailJudge
from tune.constants import TUNE_REPORT_ADD_SCHEMA, TUNE_STOPPER_DEFAULT_CHECK_INTERVAL
from tune.exceptions import TuneCompileError, TuneInterrupted
from tune.noniterative.objective import (
from tune.noniterative.stopper import NonIterativeStopper
def compute_processor(engine: ExecutionEngine, df: DataFrame) -> DataFrame:
    out_schema = df.schema + TUNE_REPORT_ADD_SCHEMA

    def get_rows() -> Iterable[Any]:
        for row in self._compute_transformer(df.as_local().as_dict_iterable(), entrypoint=entrypoint, stop_check_interval=_interval, logger=logger):
            yield [row[k] for k in out_schema.names]
    return ArrayDataFrame(get_rows(), out_schema)