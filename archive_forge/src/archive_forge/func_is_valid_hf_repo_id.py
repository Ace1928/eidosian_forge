import functools
import logging
import os
from typing import Optional
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
def is_valid_hf_repo_id(maybe_repo_id: Optional[str]) -> bool:
    """
    Check if the given string is a valid HuggingFace repo identifier e.g. "username/repo_id".
    """
    if not maybe_repo_id or os.path.isdir(maybe_repo_id):
        return False
    try:
        from huggingface_hub.utils import HFValidationError, validate_repo_id
    except ImportError:
        raise MlflowException('Unable to validate the repository identifier for the HuggingFace model hub because the `huggingface-hub` package is not installed. Please install the package with `pip install huggingface-hub` command and retry.')
    try:
        validate_repo_id(maybe_repo_id)
        return True
    except HFValidationError as e:
        _logger.warning(f'The repository identified {maybe_repo_id} is invalid: {e}')
        return False