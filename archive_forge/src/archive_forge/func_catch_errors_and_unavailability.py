from __future__ import absolute_import, unicode_literals
import functools
import logging
from ..errors import (FatalClientError, OAuth2Error, ServerError,
def catch_errors_and_unavailability(f):

    @functools.wraps(f)
    def wrapper(endpoint, uri, *args, **kwargs):
        if not endpoint.available:
            e = TemporarilyUnavailableError()
            log.info('Endpoint unavailable, ignoring request %s.' % uri)
            return ({}, e.json, 503)
        if endpoint.catch_errors:
            try:
                return f(endpoint, uri, *args, **kwargs)
            except OAuth2Error:
                raise
            except FatalClientError:
                raise
            except Exception as e:
                error = ServerError()
                log.warning('Exception caught while processing request, %s.' % e)
                return ({}, error.json, 500)
        else:
            return f(endpoint, uri, *args, **kwargs)
    return wrapper