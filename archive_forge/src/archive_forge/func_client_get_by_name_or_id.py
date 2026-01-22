from __future__ import annotations
from contextlib import contextmanager
from ansible.module_utils.basic import missing_required_lib
from .vendor.hcloud import APIException, Client as ClientBase
def client_get_by_name_or_id(client: Client, resource: str, param: str | int):
    """
    Get a resource by name, and if not found by its ID.

    :param client: Client to use to make the call
    :param resource: Name of the resource client that implements both `get_by_name` and `get_by_id` methods
    :param param: Name or ID of the resource to query
    """
    resource_client = getattr(client, resource)
    result = resource_client.get_by_name(param)
    if result is not None:
        return result
    try:
        int(param)
    except ValueError as exception:
        raise _client_resource_not_found(resource, param) from exception
    try:
        return resource_client.get_by_id(param)
    except APIException as exception:
        if exception.code == 'not_found':
            raise _client_resource_not_found(resource, param) from exception
        raise exception