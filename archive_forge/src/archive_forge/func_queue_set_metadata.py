import json
import zaqarclient.transport.errors as errors
def queue_set_metadata(transport, request, name, metadata, callback=None):
    """Sets queue metadata."""
    request.operation = 'queue_set_metadata'
    request.params['queue_name'] = name
    request.content = json.dumps(metadata)
    transport.send(request)