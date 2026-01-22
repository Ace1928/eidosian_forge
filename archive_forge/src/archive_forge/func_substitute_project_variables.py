import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import srsly
from click import NoSuchOption
from click.parser import split_arg_string
from confection import Config
from wasabi import msg
from ..cli.main import PROJECT_FILE
from ..schemas import ProjectConfigSchema, validate
from .environment import ENV_VARS
from .frozen import SimpleFrozenDict
from .logging import logger
from .validation import show_validation_error, validate_project_commands
def substitute_project_variables(config: Dict[str, Any], overrides: Dict[str, Any]=SimpleFrozenDict(), key: str='vars', env_key: str='env') -> Dict[str, Any]:
    """Interpolate variables in the project file using the config system.

    config (Dict[str, Any]): The project config.
    overrides (Dict[str, Any]): Optional config overrides.
    key (str): Key containing variables in project config.
    env_key (str): Key containing environment variable mapping in project config.
    RETURNS (Dict[str, Any]): The interpolated project config.
    """
    config.setdefault(key, {})
    config.setdefault(env_key, {})
    for config_var, env_var in config[env_key].items():
        config[env_key][config_var] = _parse_override(os.environ.get(env_var, ''))
    cfg = Config({'project': config, key: config[key], env_key: config[env_key]})
    cfg = Config().from_str(cfg.to_str(), overrides=overrides)
    interpolated = cfg.interpolate()
    return dict(interpolated['project'])