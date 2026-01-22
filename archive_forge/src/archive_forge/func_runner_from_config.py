from typing import Any, Dict, Optional
import wandb
from wandb.apis.internal import Api
from wandb.docker import is_docker_installed
from wandb.sdk.launch.errors import LaunchError
from .builder.abstract import AbstractBuilder
from .environment.abstract import AbstractEnvironment
from .registry.abstract import AbstractRegistry
from .runner.abstract import AbstractRunner
def runner_from_config(runner_name: str, api: Api, runner_config: Dict[str, Any], environment: AbstractEnvironment, registry: AbstractRegistry) -> AbstractRunner:
    """Create a runner from a config.

    This helper function is used to create a runner from a config. The
    config should have a "type" key that specifies the type of runner to import
    and create. The remaining keys are passed to the runner's from_config
    method. If the config is None or empty, a LocalContainerRunner is returned.

    Arguments:
        runner_name (str): The name of the backend.
        api (Api): The API.
        runner_config (Dict[str, Any]): The backend config.

    Returns:
        The runner.

    Raises:
        LaunchError: If the runner is not configured correctly.
    """
    if not runner_name or runner_name in ['local-container', 'local']:
        from .runner.local_container import LocalContainerRunner
        return LocalContainerRunner(api, runner_config, environment, registry)
    if runner_name == 'local-process':
        from .runner.local_process import LocalProcessRunner
        return LocalProcessRunner(api, runner_config)
    if runner_name == 'sagemaker':
        from .environment.aws_environment import AwsEnvironment
        if not isinstance(environment, AwsEnvironment):
            try:
                environment = AwsEnvironment.from_default()
            except LaunchError as e:
                raise LaunchError('Could not create Sagemaker runner. Environment must be an instance of AwsEnvironment.') from e
        from .runner.sagemaker_runner import SageMakerRunner
        return SageMakerRunner(api, runner_config, environment, registry)
    if runner_name in ['vertex', 'gcp-vertex']:
        from .environment.gcp_environment import GcpEnvironment
        if not isinstance(environment, GcpEnvironment):
            try:
                environment = GcpEnvironment.from_default()
            except LaunchError as e:
                raise LaunchError('Could not create Vertex runner. Environment must be an instance of GcpEnvironment.') from e
        from .runner.vertex_runner import VertexRunner
        return VertexRunner(api, runner_config, environment, registry)
    if runner_name == 'kubernetes':
        from .runner.kubernetes_runner import KubernetesRunner
        return KubernetesRunner(api, runner_config, environment, registry)
    raise LaunchError(f'Could not create runner from config. Invalid runner name: {runner_name}')