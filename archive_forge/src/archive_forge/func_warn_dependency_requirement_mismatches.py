import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from collections import namedtuple
from itertools import chain, filterfalse
from pathlib import Path
from threading import Timer
from typing import List, NamedTuple, Optional
import importlib_metadata
import pkg_resources  # noqa: TID251
from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version
import mlflow
from mlflow.environment_variables import MLFLOW_REQUIREMENTS_INFERENCE_TIMEOUT
from mlflow.exceptions import MlflowException
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.autologging_utils.versioning import _strip_dev_version_suffix
from mlflow.utils.databricks_utils import (
def warn_dependency_requirement_mismatches(model_requirements: List[str]):
    """
    Inspects the model's dependencies and prints a warning if the current Python environment
    doesn't satisfy them.
    """
    _DATABRICKS_FEATURE_LOOKUP = 'databricks-feature-lookup'
    try:
        mismatch_infos = []
        for req in model_requirements:
            mismatch_info = _check_requirement_satisfied(req)
            if mismatch_info is not None:
                if mismatch_info.package_name == _DATABRICKS_FEATURE_LOOKUP:
                    continue
                mismatch_infos.append(str(mismatch_info))
        if len(mismatch_infos) > 0:
            mismatch_str = ' - ' + '\n - '.join(mismatch_infos)
            warning_msg = f"Detected one or more mismatches between the model's dependencies and the current Python environment:\n{mismatch_str}\nTo fix the mismatches, call `mlflow.pyfunc.get_model_dependencies(model_uri)` to fetch the model's environment and install dependencies using the resulting environment file."
            _logger.warning(warning_msg)
    except Exception as e:
        _logger.warning(f'Encountered an unexpected error ({e!r}) while detecting model dependency mismatches. Set logging level to DEBUG to see the full traceback.')
        _logger.debug('', exc_info=True)