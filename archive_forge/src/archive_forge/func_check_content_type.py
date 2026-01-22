import re
import sys
import warnings
def check_content_type(status, headers):
    status = check_string_type(status, 'Status')
    code = int(status.split(None, 1)[0])
    NO_MESSAGE_BODY = (204, 304)
    for name, value in headers:
        name = check_string_type(name, 'Header name')
        if name.lower() == 'content-type':
            if code not in NO_MESSAGE_BODY:
                return
            assert_(0, 'Content-Type header found in a %s response, which must not return content.' % code)
    if code not in NO_MESSAGE_BODY:
        assert_(0, 'No Content-Type header found in headers (%s)' % headers)