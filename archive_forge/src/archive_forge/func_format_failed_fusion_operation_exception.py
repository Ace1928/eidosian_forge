from __future__ import absolute_import, division, print_function
import sys
import json
import re
import traceback as trace
def format_failed_fusion_operation_exception(exception):
    """Formats failed `fusion.Operation` into a simple short form, suitable
    for Ansible error output. Returns a (message: str, body: dict) tuple."""
    op = exception.op
    http_error = exception.http_error
    if op.status != 'Failed' and (not http_error):
        raise ValueError('BUG: can only format Operation exception with .status == Failed or http_error != None')
    message = None
    code = None
    operation_name = None
    operation_id = None
    try:
        if op.status == 'Failed':
            operation_id = op.id
            error = op.error
            message = error.message
            code = error.pure_code
            if not code:
                code = error.http_code
        operation_name = op.request_type
    except Exception:
        pass
    output = ''
    if operation_name:
        operation_name = re.sub('(.)([A-Z][a-z]+)', '\\1 \\2', operation_name)
        operation_name = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', operation_name).capitalize()
        output += '{0}: '.format(operation_name)
    output += 'operation failed'
    if message:
        output += ', {0}'.format(message.replace('"', "'"))
    details = DetailsPrinter(output)
    if code:
        details.append("code: '{0}'".format(code))
    if operation_id:
        details.append("operation id: '{0}'".format(operation_id))
    if http_error:
        details.append("HTTP error: '{0}'".format(str(http_error).replace('"', "'")))
    output = details.finish()
    return output