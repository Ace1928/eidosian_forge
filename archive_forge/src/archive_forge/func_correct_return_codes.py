import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
def correct_return_codes(func):
    _status_dict = {400: 'Bad Request', 401: 'Unauthorized', 403: 'Forbidden', 404: 'Not found', 409: 'Conflict', 413: 'Over Limit', 501: 'Not Implemented', 503: 'Service Unavailable'}

    def wrapper(*args, **kwargs):
        try:
            response = func(*args, **kwargs)
        except Exception as e:
            code = None
            message = 'Unknown Error'
            request_id = 'n/a'
            if hasattr(e, 'request_id'):
                request_id = e.request_id
            if hasattr(e, 'response'):
                code = e.response.status_code
                try:
                    message = e.response.json().get('faultstring', _status_dict.get(code, message))
                except Exception:
                    message = _status_dict.get(code, message)
            elif isinstance(e, osc_exc.ClientException) and e.code != e.http_status:
                code = e.http_status
                message = e.code
            else:
                raise
            raise OctaviaClientException(code=code, message=message, request_id=request_id) from e
        return response
    return wrapper