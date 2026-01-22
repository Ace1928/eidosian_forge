import json
import zaqarclient.transport.errors as errors
def flavor_create(transport, request, name, flavor_data):
    """Creates a flavor called `name`

    :param transport: Transport instance to use
    :type transport: `transport.base.Transport`
    :param request: Request instance ready to be sent.
    :type request: `transport.request.Request`
    :param name: Flavor reference name.
    :type name: str
    :param flavor_data: Flavor's properties, i.e: pool, capabilities.
    :type flavor_data: `dict`
    """
    request.operation = 'flavor_create'
    request.params['flavor_name'] = name
    request.content = json.dumps(flavor_data)
    transport.send(request)