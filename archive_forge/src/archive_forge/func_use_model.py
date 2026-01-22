import json
import os
from typing import Any, Dict, List, Optional, Union
import wandb
import wandb.data_types as data_types
from wandb.data_types import _SavedModel
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry
def use_model(aliased_path: str) -> '_SavedModel':
    """Fetch a saved model from an alias.

    Under the hood, we use the alias to fetch the model artifact containing the
    serialized model files and rebuild the model object from these files. We also
    declare the fetched model artifact as an input to the run (with `run.use_artifact`).

    Args:
        aliased_path: `str` - the following forms are valid: "name:version",
            "name:alias". May be prefixed with "entity/project".

    Returns:
        _SavedModel instance

    Example:
        ```python
        # Assuming you have previously logged a model with the name "my-simple-model":
        sm = use_model("my-simple-model:latest")
        model = sm.model_obj()
        ```
    """
    if ':' not in aliased_path:
        raise ValueError("aliased_path must be of the form 'name:alias' or 'name:version'.")
    if wandb.run:
        run = wandb.run
        artifact = run.use_artifact(aliased_path)
        sm = artifact.get('index')
        if sm is None or not isinstance(sm, _SavedModel):
            raise ValueError('Deserialization into model object failed: _SavedModel instance could not be initialized properly.')
        return sm
    else:
        raise ValueError('use_model can only be called inside a run. Please call wandb.init() before use_model(...)')