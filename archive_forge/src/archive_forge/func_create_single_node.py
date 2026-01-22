import functools
import json
import jsonschema
import yaml
from ironicclient import exc
@create_single_handler('node')
def create_single_node(client, **params):
    """Call the client to create a node.

    :param client: ironic client instance.
    :param params: dictionary to be POSTed to /nodes endpoint, excluding
        "ports" and "portgroups" keys.
    :returns: UUID of the created node or None in case of exception, and an
        exception, if it appears.
    :raises: InvalidAttribute, if some parameters passed to client's
        create_method are invalid.
    :raises: ClientException, if the creation of the node fails.
    """
    params.pop('ports', None)
    params.pop('portgroups', None)
    ret = client.node.create(**params)
    return ret.uuid