from __future__ import unicode_literals
import re
def is_uri_reference(uri):
    return re.match(URI_reference, uri, re.VERBOSE)