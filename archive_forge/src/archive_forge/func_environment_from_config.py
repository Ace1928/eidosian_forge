from typing import Any, Dict, Optional
import wandb
from wandb.apis.internal import Api
from wandb.docker import is_docker_installed
from wandb.sdk.launch.errors import LaunchError
from .builder.abstract import AbstractBuilder
from .environment.abstract import AbstractEnvironment
from .registry.abstract import AbstractRegistry
from .runner.abstract import AbstractRunner
def environment_from_config(config: Optional[Dict[str, Any]]) -> AbstractEnvironment:
    """Create an environment from a config.

    This helper function is used to create an environment from a config. The
    config should have a "type" key that specifies the type of environment to
    create. The remaining keys are passed to the environment's from_config
    method. If the config is None or empty, a LocalEnvironment is returned.

    Arguments:
        config (Dict[str, Any]): The config.

    Returns:
        Environment: The environment constructed.
    """
    if not config:
        from .environment.local_environment import LocalEnvironment
        return LocalEnvironment()
    env_type = config.get('type')
    if not env_type:
        raise LaunchError('Could not create environment from config. Environment type not specified!')
    if env_type == 'local':
        from .environment.local_environment import LocalEnvironment
        return LocalEnvironment.from_config(config)
    if env_type == 'aws':
        from .environment.aws_environment import AwsEnvironment
        return AwsEnvironment.from_config(config)
    if env_type == 'gcp':
        from .environment.gcp_environment import GcpEnvironment
        return GcpEnvironment.from_config(config)
    if env_type == 'azure':
        from .environment.azure_environment import AzureEnvironment
        return AzureEnvironment.from_config(config)
    raise LaunchError(f'Could not create environment from config. Invalid type: {env_type}')