import logging
from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.exceptions import MlflowException
from mlflow.store.model_registry import (
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS, utils
from mlflow.utils.arguments_utils import _get_arg_names
def create_registered_model(self, name, tags=None, description=None):
    """Create a new registered model in backend store.

        Args:
            name: Name of the new model. This is expected to be unique in the backend store.
            tags: A dictionary of key-value pairs that are converted into
                :py:class:`mlflow.entities.model_registry.RegisteredModelTag` objects.
            description: Description of the model.

        Returns:
            A single object of :py:class:`mlflow.entities.model_registry.RegisteredModel`
            created by backend.

        """
    tags = tags if tags else {}
    tags = [RegisteredModelTag(key, str(value)) for key, value in tags.items()]
    return self.store.create_registered_model(name, tags, description)