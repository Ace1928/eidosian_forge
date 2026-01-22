import logging
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_STATE
from mlflow.transformers.flavor_config import FlavorKey, get_peft_base_model, is_peft_model
def load_model_and_components_from_huggingface_hub(flavor_conf, accelerate_conf, device=None):
    """
    Load the model and components of a Transformer pipeline from HuggingFace Hub.

    Args:
        flavor_conf: The flavor configuration
        accelerate_conf: The configuration for the accelerate library
        device: The device to load the model onto
    """
    loaded = {}
    model_repo = flavor_conf[FlavorKey.MODEL_NAME]
    model_revision = flavor_conf.get(FlavorKey.MODEL_REVISION)
    if not model_revision:
        raise MlflowException("The model was saved with 'save_pretrained' set to False, but the commit hash is not found in the saved metadata. Loading the model with the different version may cause inconsistency issue and security risk.", error_code=INVALID_STATE)
    loaded[FlavorKey.MODEL] = _load_model(model_repo, flavor_conf, accelerate_conf, device, revision=model_revision)
    components = flavor_conf.get(FlavorKey.COMPONENTS, [])
    if FlavorKey.PROCESSOR_TYPE in flavor_conf:
        components.append('processor')
    for name in components:
        loaded[name] = _load_component(flavor_conf, name)
    return loaded