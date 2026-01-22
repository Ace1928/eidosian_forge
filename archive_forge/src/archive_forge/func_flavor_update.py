import json
import zaqarclient.transport.errors as errors
def flavor_update(transport, request, flavor_name, flavor_data):
    """Updates the flavor `flavor_name`

    :param transport: Transport instance to use
    :type transport: `transport.base.Transport`
    :param request: Request instance ready to be sent.
    :type request: `transport.request.Request`
    :param flavor_name: Flavor reference name.
    :type flavor_name: str
    :param flavor_data: Flavor's properties, i.e: pool, capabilities.
    :type flavor_data: `dict`
    """
    request.operation = 'flavor_update'
    request.params['flavor_name'] = flavor_name
    request.content = json.dumps(flavor_data)
    resp = transport.send(request)
    return resp.deserialized_content