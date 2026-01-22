import textwrap
import warnings
from functools import wraps
from typing import Dict
import importlib_metadata
from packaging.version import Version
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.utils.autologging_utils.versioning import (
imports declared from a common root path if multiple files are defined with import dependencies
@wraps(func)
def version_func(*args, **kwargs):
    installed_version = Version(importlib_metadata.version(module_key))
    if installed_version < Version(min_ver) or installed_version > Version(max_ver):
        warnings.warn(notice, category=FutureWarning, stacklevel=2)
    return func(*args, **kwargs)