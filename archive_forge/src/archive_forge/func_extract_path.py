import os.path
import logging
from wsme.utils import OrderedDict
from wsme.protocol import CallContext, Protocol, media_type_accept
import wsme.rest
from wsme.rest import json
from wsme.rest import xml
import wsme.runtime
def extract_path(self, context):
    path = context.request.path
    assert path.startswith(self.root._webpath)
    path = path[len(self.root._webpath):]
    path = path.strip('/').split('/')
    for dataformat in self.dataformats:
        if path[-1].endswith('.' + dataformat):
            path[-1] = path[-1][:-len(dataformat) - 1]
    for p, fdef in self.root.getapi():
        if p == path:
            return path
    for p, fdef in self.root.getapi():
        if len(p) == len(path) + 1 and p[:len(path)] == path and (fdef.extra_options.get('method') == context.request.method):
            return p
    return path