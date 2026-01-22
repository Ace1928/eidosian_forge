import importlib
import re
from packaging.version import InvalidVersion, Version
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.utils.databricks_utils import is_in_databricks_runtime
def get_min_max_version_and_pip_release(module_key):
    min_version = _ML_PACKAGE_VERSIONS[module_key]['autologging']['minimum']
    max_version = _ML_PACKAGE_VERSIONS[module_key]['autologging']['maximum']
    pip_release = _ML_PACKAGE_VERSIONS[module_key]['package_info']['pip_release']
    return (min_version, max_version, pip_release)