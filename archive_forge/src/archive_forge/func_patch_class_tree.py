import inspect
import itertools
import logging
import os
from typing import Any, Dict, Optional
import yaml
import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.autologging_utils import (
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.validation import _is_numeric
def patch_class_tree(klass):
    """
        Patches all subclasses that override any auto-loggable method via monkey patching using
        the gorilla package, taking the argument as the tree root in the class hierarchy. Every
        auto-loggable method found in any of the subclasses is replaced by the patched version.

        Args:
            klass: Root in the class hierarchy to be analyzed and patched recursively.
        """
    autolog_supported_func = {'fit': wrapper_fit}
    glob_subclasses = set(find_subclasses(klass))
    patches_list = [(clazz, method_name, wrapper_func) for clazz in glob_subclasses for method_name, wrapper_func in autolog_supported_func.items() if overrides(clazz, method_name)]
    for clazz, method_name, patch_impl in patches_list:
        safe_patch(FLAVOR_NAME, clazz, method_name, patch_impl, manage_run=True, extra_tags=extra_tags)