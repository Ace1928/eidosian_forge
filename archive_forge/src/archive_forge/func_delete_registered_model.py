import logging
from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.exceptions import MlflowException
from mlflow.store.model_registry import (
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS, utils
from mlflow.utils.arguments_utils import _get_arg_names
def delete_registered_model(self, name):
    """Delete registered model.
        Backend raises exception if a registered model with given name does not exist.

        Args:
            name: Name of the registered model to delete.
        """
    self.store.delete_registered_model(name)