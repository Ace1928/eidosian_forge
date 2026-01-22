from typing import Dict, Any, Tuple, Callable, List, Optional, IO
from types import ModuleType
import os
import sys
from spacy import Language, load
from spacy.util import SimpleFrozenList
from .util import dict_to_dot, dot_to_dict, matcher_for_regex_patterns
from .util import setup_default_console_logger, LoggerT
def mlflow_logger_v2(run_id: Optional[str]=None, experiment_id: Optional[str]=None, run_name: Optional[str]=None, nested: bool=False, tags: Optional[Dict[str, Any]]=None, remove_config_values: List[str]=SimpleFrozenList(), log_custom_stats: Optional[List[str]]=None) -> LoggerT:
    """Creates a logger that interoperates with the MLflow framework.

    Args:
        run_id (Optional[str]):
            Unique ID of an existing MLflow run to which parameters and metrics are logged. Can be omitted if `experiment_id` and `run_id` are provided. Defaults to None.
        experiment_id (Optional[str]):
            ID of an existing experiment under which to create the current run. Only applicable when `run_id` is `None`. Defaults to None.
        run_name (Optional[str]):
            Name of new run. Only applicable when `run_id` is `None`. Defaults to None.
        nested (bool):
            Controls whether run is nested in parent run. `True` creates a nested run. Defaults to False.
        tags (Optional[Dict[str, Any]]):
            A dictionary of string keys and values to set as tags on the run. If a run is being resumed, these tags are set on the resumed run. If a new run is being created, these tags are set on the new run. Defaults to None.
        remove_config_values (List[str]):
            A list of values to exclude from the config before it is uploaded to MLflow. Defaults to an empty list.
        log_custom_stats (Optional[List[str]]):
            A list of regular expressions that will be applied to the info dictionary passed to the logger. Statistics and metrics that match these regexps will be automatically logged. Defaults to None.

    Returns:
        LoggerT: Logger instance.
    """
    mlflow = _import_mlflow()

    def setup_logger(nlp: Language, stdout: IO=sys.stdout, stderr: IO=sys.stderr) -> Tuple[Callable[[Dict[str, Any]], None], Callable[[], None]]:
        match_stat = matcher_for_regex_patterns(log_custom_stats)
        _setup_mlflow(mlflow, nlp, run_id, experiment_id, run_name, nested, tags, remove_config_values)

        def log_step(info: Optional[Dict[str, Any]]):
            _log_step_mlflow(mlflow, info)
            _log_custom_stats(mlflow, info, match_stat)

        def finalize() -> None:
            _finalize_mlflow(mlflow)
        return (log_step, finalize)
    return setup_logger