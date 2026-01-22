import sys
from .recursive import ForwardRequestException, RecursionLoop
def replacement_start_response(status, headers, exc_info=None):
    """
            Overrides the default response if the status is defined in the
            Pecan app error map configuration.
            """
    try:
        status_code = int(status.split(' ')[0])
    except (ValueError, TypeError):
        raise Exception('ErrorDocumentMiddleware received an invalid status %s' % status)
    if status_code in self.error_map:

        def factory(app):
            return StatusPersist(app, status, self.error_map[status_code])
        raise ForwardRequestException(factory=factory)
    return start_response(status, headers, exc_info)