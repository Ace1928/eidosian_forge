from typing import Any, Dict, Optional
import wandb
from wandb.apis.internal import Api
from wandb.docker import is_docker_installed
from wandb.sdk.launch.errors import LaunchError
from .builder.abstract import AbstractBuilder
from .environment.abstract import AbstractEnvironment
from .registry.abstract import AbstractRegistry
from .runner.abstract import AbstractRunner
def registry_from_config(config: Optional[Dict[str, Any]], environment: AbstractEnvironment) -> AbstractRegistry:
    """Create a registry from a config.

    This helper function is used to create a registry from a config. The
    config should have a "type" key that specifies the type of registry to
    create. The remaining keys are passed to the registry's from_config
    method. If the config is None or empty, a LocalRegistry is returned.

    Arguments:
        config (Dict[str, Any]): The registry config.
        environment (Environment): The environment of the registry.

    Returns:
        The registry if config is not None, otherwise None.

    Raises:
        LaunchError: If the registry is not configured correctly.
    """
    if not config:
        from .registry.local_registry import LocalRegistry
        return LocalRegistry()
    wandb.termwarn('The `registry` block of the launch agent config is being deprecated. Please specify an image repository URI under the `builder.destination` key of your launch agent config. See https://docs.wandb.ai/guides/launch/setup-agent-advanced#agent-configuration for more information.')
    registry_type = config.get('type')
    if registry_type is None or registry_type == 'local':
        from .registry.local_registry import LocalRegistry
        return LocalRegistry()
    if registry_type == 'ecr':
        from .registry.elastic_container_registry import ElasticContainerRegistry
        return ElasticContainerRegistry.from_config(config)
    if registry_type == 'gcr':
        from .registry.google_artifact_registry import GoogleArtifactRegistry
        return GoogleArtifactRegistry.from_config(config)
    if registry_type == 'acr':
        from .registry.azure_container_registry import AzureContainerRegistry
        return AzureContainerRegistry.from_config(config)
    raise LaunchError(f'Could not create registry from config. Invalid registry type: {registry_type}')