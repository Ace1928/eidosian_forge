import abc
import json
import logging
import os
import time
import traceback
from enum import Enum
from typing import Any, Dict, List, Optional
import yaml
from mlflow.recipes.cards import CARD_HTML_NAME, CARD_PICKLE_NAME, BaseCard, FailureCard
from mlflow.recipes.utils import get_recipe_name
from mlflow.recipes.utils.step import display_html
from mlflow.tracking import MlflowClient
from mlflow.utils.databricks_utils import is_in_databricks_runtime
@classmethod
def from_step_config_path(cls, step_config_path: str, recipe_root: str) -> 'BaseStep':
    """
        Constructs a step class instance using the config specified in the
        configuration file.

        Args:
            step_config_path: String path to the step-specific configuration
                on the local filesystem.
            recipe_root: String path to the recipe root directory on
                the local filesystem.

        Returns:
            class instance of the step.
        """
    with open(step_config_path) as f:
        step_config = yaml.safe_load(f)
    return cls(step_config, recipe_root)