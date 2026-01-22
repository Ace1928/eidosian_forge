import json
import zaqarclient.transport.errors as errors
def queue_get_metadata(transport, request, name, callback=None):
    """Gets queue metadata."""
    return _common_queue_ops('queue_get_metadata', transport, request, name, callback=callback)