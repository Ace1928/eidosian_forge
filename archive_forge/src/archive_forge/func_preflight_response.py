import functools
import re
import typing
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import PlainTextResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send
def preflight_response(self, request_headers: Headers) -> Response:
    requested_origin = request_headers['origin']
    requested_method = request_headers['access-control-request-method']
    requested_headers = request_headers.get('access-control-request-headers')
    headers = dict(self.preflight_headers)
    failures = []
    if self.is_allowed_origin(origin=requested_origin):
        if self.preflight_explicit_allow_origin:
            headers['Access-Control-Allow-Origin'] = requested_origin
    else:
        failures.append('origin')
    if requested_method not in self.allow_methods:
        failures.append('method')
    if self.allow_all_headers and requested_headers is not None:
        headers['Access-Control-Allow-Headers'] = requested_headers
    elif requested_headers is not None:
        for header in [h.lower() for h in requested_headers.split(',')]:
            if header.strip() not in self.allow_headers:
                failures.append('headers')
                break
    if failures:
        failure_text = 'Disallowed CORS ' + ', '.join(failures)
        return PlainTextResponse(failure_text, status_code=400, headers=headers)
    return PlainTextResponse('OK', status_code=200, headers=headers)