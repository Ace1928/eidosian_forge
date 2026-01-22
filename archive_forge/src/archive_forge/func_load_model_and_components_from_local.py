import logging
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_STATE
from mlflow.transformers.flavor_config import FlavorKey, get_peft_base_model, is_peft_model
def load_model_and_components_from_local(path, flavor_conf, accelerate_conf, device=None):
    """
    Load the model and components of a Transformer pipeline from the specified local path.

    Args:
        path: The local path contains MLflow model artifacts
        flavor_conf: The flavor configuration
        accelerate_conf: The configuration for the accelerate library
        device: The device to load the model onto
    """
    loaded = {}
    model_path = path.joinpath(flavor_conf.get(FlavorKey.MODEL_BINARY, 'pipeline'))
    loaded[FlavorKey.MODEL] = _load_model(model_path, flavor_conf, accelerate_conf, device)
    components = flavor_conf.get(FlavorKey.COMPONENTS, [])
    if FlavorKey.PROCESSOR_TYPE in flavor_conf:
        components.append('processor')
    for component_key in components:
        loaded[component_key] = _load_component(flavor_conf, component_key, local_path=path)
    return loaded