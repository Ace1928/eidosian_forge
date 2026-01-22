import json
import zaqarclient.transport.errors as errors
def flavor_list(transport, request, **kwargs):
    """Gets a list of flavors

    :param transport: Transport instance to use
    :type transport: `transport.base.Transport`
    :param request: Request instance ready to be sent.
    :type request: `transport.request.Request`
    :param kwargs: Optional arguments for this operation.
        - marker: Where to start getting flavors from.
        - limit: Maximum number of flavors to get.
    """
    request.operation = 'flavor_list'
    request.params.update(kwargs)
    resp = transport.send(request)
    if not resp.content:
        return {'links': [], 'flavors': []}
    return resp.deserialized_content