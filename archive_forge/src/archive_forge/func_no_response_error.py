from __future__ import (absolute_import, division, print_function)
def no_response_error(api, response):
    """format error message for empty response"""
    return 'calling: %s: no response %s.' % (api, repr(response))