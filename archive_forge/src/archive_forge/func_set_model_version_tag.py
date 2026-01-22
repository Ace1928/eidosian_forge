import logging
from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.exceptions import MlflowException
from mlflow.store.model_registry import (
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS, utils
from mlflow.utils.arguments_utils import _get_arg_names
def set_model_version_tag(self, name, version, key, value):
    """Set a tag for the model version.

        Args:
            name: Registered model name.
            version: Registered model version.
            key: Tag key to log.
            value: Tag value to log.

        Returns:
            None
        """
    self.store.set_model_version_tag(name, version, ModelVersionTag(key, str(value)))