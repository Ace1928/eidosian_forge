import os
import posixpath
import re
def autohandler(template, context, name='autohandler'):
    lookup = context.lookup
    _template_uri = template.module._template_uri
    if not lookup.filesystem_checks:
        try:
            return lookup._uri_cache[autohandler, _template_uri, name]
        except KeyError:
            pass
    tokens = re.findall('([^/]+)', posixpath.dirname(_template_uri)) + [name]
    while len(tokens):
        path = '/' + '/'.join(tokens)
        if path != _template_uri and _file_exists(lookup, path):
            if not lookup.filesystem_checks:
                return lookup._uri_cache.setdefault((autohandler, _template_uri, name), path)
            else:
                return path
        if len(tokens) == 1:
            break
        tokens[-2:] = [name]
    if not lookup.filesystem_checks:
        return lookup._uri_cache.setdefault((autohandler, _template_uri, name), None)
    else:
        return None