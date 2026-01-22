import re
import jmespath
from botocore import xform_name
from ..exceptions import ResourceLoadException
def build_param_structure(params, target, value, index=None):
    """
    This method provides a basic reverse JMESPath implementation that
    lets you go from a JMESPath-like string to a possibly deeply nested
    object. The ``params`` are mutated in-place, so subsequent calls
    can modify the same element by its index.

        >>> build_param_structure(params, 'test[0]', 1)
        >>> print(params)
        {'test': [1]}

        >>> build_param_structure(params, 'foo.bar[0].baz', 'hello world')
        >>> print(params)
        {'test': [1], 'foo': {'bar': [{'baz': 'hello, world'}]}}

    """
    pos = params
    parts = target.split('.')
    for i, part in enumerate(parts):
        result = INDEX_RE.search(part)
        if result:
            if result.group(1):
                if result.group(1) == '*':
                    part = part[:-3]
                else:
                    index = int(result.group(1))
                    part = part[:-len(str(index) + '[]')]
            else:
                index = None
                part = part[:-2]
            if part not in pos or not isinstance(pos[part], list):
                pos[part] = []
            if index is None:
                index = len(pos[part])
            while len(pos[part]) <= index:
                pos[part].append({})
            if i == len(parts) - 1:
                pos[part][index] = value
            else:
                pos = pos[part][index]
        else:
            if part not in pos:
                pos[part] = {}
            if i == len(parts) - 1:
                pos[part] = value
            else:
                pos = pos[part]