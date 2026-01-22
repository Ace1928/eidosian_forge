from typing import Dict, Any, Tuple, Callable, List, Optional, IO
from types import ModuleType
import os
import sys
from spacy import Language
from spacy.util import SimpleFrozenList
from .util import dict_to_dot, dot_to_dict, matcher_for_regex_patterns
from .util import setup_default_console_logger, LoggerT
def clearml_logger_v2(project_name: str, task_name: str, remove_config_values: List[str]=SimpleFrozenList(), model_log_interval: Optional[int]=None, log_dataset_dir: Optional[str]=None, log_best_dir: Optional[str]=None, log_latest_dir: Optional[str]=None, log_custom_stats: Optional[List[str]]=None) -> LoggerT:
    """Creates a logger that interoperates with the ClearML framework.

    Args:
        project_name (str):
            The name of the project in the ClearML interface. The project will be created automatically if it doesn't exist yet.
        task_name (str):
            The name of the ClearML task. A task is an experiment that lives inside a project. Can be non-unique.
        remove_config_values (List[str]):
            A list of values to exclude from the config before it is uploaded to ClearML. Defaults to [].
        model_log_interval (Optional[int]):
            Steps to wait between logging model checkpoints to the ClearML dasboard (default: `None`). Will have no effect without also setting `log_best_dir` or `log_latest_dir`. Defaults to None.
        log_dataset_dir (Optional[str]):
            Directory containing the dataset to be logged and versioned as a ClearML Dataset. Defaults to None.
        log_best_dir (Optional[str]):
            Directory containing the best trained model as saved by spaCy, to be logged and versioned as a ClearML artifact. Defaults to None.
        log_latest_dir (Optional[str]):
            Directory containing the latest trained model as saved by spaCy, to be logged and versioned as a ClearML artifact. Defaults to None.
        log_custom_stats (Optional[List[str]]):
            A list of regular expressions that will be applied to the info dictionary passed to the logger. Statistics and metrics that match these regexps will be automatically logged. Defaults to None.

    Returns:
        LoggerT: Logger instance.
    """
    clearml = _import_clearml()

    def setup_logger(nlp: Language, stdout: IO=sys.stdout, stderr: IO=sys.stderr) -> Tuple[Callable[[Dict[str, Any]], None], Callable[[], None]]:
        match_stat = matcher_for_regex_patterns(log_custom_stats)
        task, best_model, last_model = _setup_clearml(clearml, nlp, project_name, task_name, log_dataset_dir, log_best_dir, log_latest_dir, remove_config_values)

        def log_step(info: Optional[Dict[str, Any]]):
            _log_step_clearml(info, task, best_model, last_model, model_log_interval, log_best_dir, log_latest_dir)
            _log_custom_stats(clearml, info, match_stat)

        def finalize():
            _finalize_clearml(task)
        return (log_step, finalize)
    return setup_logger