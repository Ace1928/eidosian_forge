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
@click.argument('resource', type=click.Choice(_get_available_resources()))
@click.option('--format', default='default', type=click.Choice(_get_available_formats()))
@click.option('-f', '--filter', help="A key, predicate, and value to filter the result. E.g., --filter 'key=value' or --filter 'key!=value'. You can specify multiple --filter options. In this case all predicates are concatenated as AND. For example, --filter key=value --filter key2=value means (key==val) AND (key2==val2), String filter values are case-insensitive.", multiple=True)
@click.option('--limit', default=DEFAULT_LIMIT, type=int, help='Maximum number of entries to return. 100 by default.')
@click.option('--detail', help='If the flag is set, the output will contain data in more details. Note that the API could query more sources to obtain information in a greater detail.', is_flag=True, default=False)
@timeout_option
@address_option
@PublicAPI(stability='stable')
def ray_list(resource: str, format: str, filter: List[str], limit: int, detail: bool, timeout: float, address: str):
    """List all states of a given resource.

    Normally, summary APIs are recommended before listing all resources.

    The output schema is defined at :ref:`State API Schema section. <state-api-schema>`

    For example, the output schema of `ray list tasks` is
    :class:`~ray.util.state.common.TaskState`.

    Usage:

        List all actor information from the cluster.

        ```
        ray list actors
        ```

        List 50 actors from the cluster. The sorting order cannot be controlled.

        ```
        ray list actors --limit 50
        ```

        List 10 actors with state PENDING.

        ```
        ray list actors --limit 10 --filter "state=PENDING"
        ```

        List actors with yaml format.

        ```
        ray list actors --format yaml
        ```

        List actors with details. When --detail is specified, it might query
        more data sources to obtain data in details.

        ```
        ray list actors --detail
        ```

    The API queries one or more components from the cluster to obtain the data.
    The returned state snapshot could be stale, and it is not guaranteed to return
    the live data.

    The API can return partial or missing output upon the following scenarios.

    - When the API queries more than 1 component, if some of them fail,
      the API will return the partial result (with a suppressible warning).
    - When the API returns too many entries, the API
      will truncate the output. Currently, truncated data cannot be
      selected by users.

    Args:
        resource: The type of the resource to query.

    Raises:
        :class:`RayStateApiException <ray.util.state.exception.RayStateApiException>`
            if the CLI is failed to query the data.

    Changes:
        - changed in version 2.7: --filter values are case-insensitive.

    """
    resource = StateResource(resource.replace('-', '_'))
    format = AvailableFormat(format)
    client = StateApiClient(address=address)
    filter = [_parse_filter(f) for f in filter]
    options = ListApiOptions(limit=limit, timeout=timeout, filters=filter, detail=detail)
    try:
        data = client.list(resource, options=options, raise_on_missing_output=False, _explain=_should_explain(format))
    except RayStateApiException as e:
        raise click.UsageError(str(e))
    if detail and format == AvailableFormat.DEFAULT:
        format = AvailableFormat.YAML
    print(format_list_api_output(state_data=data, schema=resource_to_schema(resource), format=format, detail=detail))