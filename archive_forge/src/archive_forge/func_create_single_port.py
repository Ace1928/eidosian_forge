import functools
import json
import jsonschema
import yaml
from ironicclient import exc
@create_single_handler('port')
def create_single_port(client, **params):
    """Call the client to create a port.

    :param client: ironic client instance.
    :param params: dictionary to be POSTed to /ports endpoint.
    :returns: UUID of the created port or None in case of exception, and an
        exception, if it appears.
    :raises: InvalidAttribute, if some parameters passed to client's
        create_method are invalid.
    :raises: ClientException, if the creation of the port fails.
    """
    ret = client.port.create(**params)
    return ret.uuid