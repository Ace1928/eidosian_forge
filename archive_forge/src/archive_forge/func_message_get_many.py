import json
import zaqarclient.transport.errors as errors
def message_get_many(transport, request, queue_name, messages, callback=None):
    """Gets many messages by id

    :param transport: Transport instance to use
    :type transport: `transport.base.Transport`
    :param request: Request instance ready to be sent.
    :type request: `transport.request.Request`
    :param queue_name: Queue reference name.
    :type queue_name: str
    :param messages: Messages references.
    :param messages: list of str
    :param callback: Optional callable to use as callback.
        If specified, this request will be sent asynchronously.
        (IGNORED UNTIL ASYNC SUPPORT IS COMPLETE)
    :type callback: Callable object.
    """
    request.operation = 'message_get_many'
    request.params['queue_name'] = queue_name
    request.params['ids'] = messages
    resp = transport.send(request)
    return resp.deserialized_content