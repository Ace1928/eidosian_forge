import logging
from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.exceptions import MlflowException
from mlflow.store.model_registry import (
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS, utils
from mlflow.utils.arguments_utils import _get_arg_names
def get_model_version_download_uri(self, name, version):
    """Get the download location in Model Registry for this model version.

        Args:
            name: Name of the containing registered model.
            version: Version number of the model version.

        Returns:
            A single URI location that allows reads for downloading.

        """
    return self.store.get_model_version_download_uri(name, version)