import itertools
import json
import pkgutil
import re
from jsonschema.compat import str_types, MutableMapping, urlsplit
def extras_msg(extras):
    """
    Create an error message for extra items or properties.

    """
    if len(extras) == 1:
        verb = 'was'
    else:
        verb = 'were'
    return (', '.join((repr(extra) for extra in extras)), verb)