import json
import logging
from datetime import datetime
from enum import Enum, unique
from typing import Dict, List, Optional, Tuple
import click
import yaml
import ray._private.services as services
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.util.state import (
from ray.util.state.common import (
from ray.util.state.exception import RayStateApiException
from ray.util.annotations import PublicAPI
@click.command()
@click.argument('resource', type=click.Choice(_get_available_resources(excluded=[StateResource.JOBS, StateResource.RUNTIME_ENVS])))
@click.argument('id', type=str)
@address_option
@timeout_option
@PublicAPI(stability='stable')
def ray_get(resource: str, id: str, address: Optional[str], timeout: float):
    """Get a state of a given resource by ID.

    We currently DO NOT support get by id for jobs and runtime-envs

    The output schema is defined at :ref:`State API Schema section. <state-api-schema>`

    For example, the output schema of `ray get tasks <task-id>` is
    :class:`~ray.util.state.common.TaskState`.

    Usage:

        Get an actor with actor id <actor-id>

        ```
        ray get actors <actor-id>
        ```

        Get a placement group information with <placement-group-id>

        ```
        ray get placement-groups <placement-group-id>
        ```

    The API queries one or more components from the cluster to obtain the data.
    The returned state snapshot could be stale, and it is not guaranteed to return
    the live data.

    Args:
        resource: The type of the resource to query.
        id: The id of the resource.

    Raises:
        :class:`RayStateApiException <ray.util.state.exception.RayStateApiException>`
            if the CLI is failed to query the data.
    """
    resource = StateResource(resource.replace('-', '_'))
    logger.debug(f'Create StateApiClient to ray instance at: {address}...')
    client = StateApiClient(address=address)
    options = GetApiOptions(timeout=timeout)
    try:
        data = client.get(resource=resource, id=id, options=options, _explain=_should_explain(AvailableFormat.YAML))
    except RayStateApiException as e:
        raise click.UsageError(str(e))
    print(format_get_api_output(state_data=data, id=id, schema=resource_to_schema(resource), format=AvailableFormat.YAML))