from __future__ import absolute_import, division, print_function
import sys
import json
import re
import traceback as trace
def format_http_exception(exception, traceback):
    """Formats failed `urllib3.exceptions` exceptions into a simple short form,
    suitable for Ansible error output. Returns a `str`."""
    output = ''
    call_site = _extract_rest_call_site(traceback)
    if call_site:
        output += "'{0}': ".format(call_site)
    output += 'HTTP request failed via '
    inner = exception
    while True:
        try:
            e = inner.reason
            if e and isinstance(e, urllib3.exceptions.HTTPError):
                inner = e
                continue
            break
        except Exception:
            break
    if inner != exception:
        output += "'{0}'/'{1}'".format(type(inner).__name__, type(exception).__name__)
    else:
        output += "'{0}'".format(type(exception).__name__)
    output += ' - {0}'.format(str(exception).replace('"', "'"))
    return output