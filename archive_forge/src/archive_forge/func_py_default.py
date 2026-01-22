import re
from collections import namedtuple
def py_default(type_name):
    """Get the Python default value for a given model type.

        >>> py_default('string')
        ''string''
        >>> py_default('list')
        '[...]'
        >>> py_default('unknown')
        '...'

    :rtype: string
    """
    return {'double': '123.0', 'long': '123', 'integer': '123', 'string': "'string'", 'blob': "b'bytes'", 'boolean': 'True|False', 'list': '[...]', 'map': '{...}', 'structure': '{...}', 'timestamp': 'datetime(2015, 1, 1)'}.get(type_name, '...')