import importlib
import json
import os
from pathlib import Path
import re
import sys
import typer
from typing import Optional
import uuid
import yaml
import ray
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.resources import resources_to_json, json_to_resources
from ray.tune.tune import run_experiments
from ray.tune.schedulers import create_scheduler
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.common import CLIArguments as cli
from ray.rllib.common import FrameworkEnum, SupportedFileType
from ray.rllib.common import download_example_file, get_file_type
def load_experiments_from_file(config_file: str, file_type: SupportedFileType, stop: Optional[str]=None, checkpoint_config: Optional[dict]=None) -> dict:
    """Load experiments from a file. Supports YAML and Python files.

    If you want to use a Python file, it has to have a 'config' variable
    that is an AlgorithmConfig object and - optionally - a `stop` variable defining
    the stop criteria.

    Args:
        config_file: The yaml or python file to be used as experiment definition.
            Must only contain exactly one experiment.
        file_type: One value of the `SupportedFileType` enum (yaml or python).
        stop: An optional stop json string, only used if file_type is python.
            If None (and file_type is python), will try to extract stop information
            from a defined `stop` variable in the python file, otherwise, will use {}.
        checkpoint_config: An optional checkpoint config to add to the returned
            experiments dict.

    Returns:
        The experiments dict ready to be passed into `tune.run_experiments()`.
    """
    if file_type == SupportedFileType.yaml:
        with open(config_file) as f:
            experiments = yaml.safe_load(f)
            if stop is not None and stop != '{}':
                raise ValueError('`stop` criteria only supported for python files.')
    else:
        module_name = os.path.basename(config_file).replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, config_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        if not hasattr(module, 'config'):
            raise ValueError("Your Python file must contain a 'config' variable that is an AlgorithmConfig object.")
        algo_config = getattr(module, 'config')
        if stop is None:
            stop = getattr(module, 'stop', {})
        else:
            stop = json.loads(stop)
        config = algo_config.to_dict()
        experiments = {f'default_{uuid.uuid4().hex}': {'run': algo_config.algo_class, 'env': config.get('env'), 'config': config, 'stop': stop}}
    for key, val in experiments.items():
        experiments[key]['checkpoint_config'] = checkpoint_config or {}
    return experiments