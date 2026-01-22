import logging
from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.exceptions import MlflowException
from mlflow.store.model_registry import (
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS, utils
from mlflow.utils.arguments_utils import _get_arg_names
def transition_model_version_stage(self, name, version, stage, archive_existing_versions=False):
    """Update model version stage.

        Args:
            name: Registered model name.
            version: Registered model version.
            stage: New desired stage for this model version.
            archive_existing_versions: If this flag is set to ``True``, all existing model
                versions in the stage will be automatically moved to the "archived" stage. Only
                valid when ``stage`` is ``"staging"`` or ``"production"`` otherwise an error will be
                raised.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.

        """
    if stage.strip() == '':
        raise MlflowException('The stage must not be an empty string.')
    return self.store.transition_model_version_stage(name=name, version=version, stage=stage, archive_existing_versions=archive_existing_versions)