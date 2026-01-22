import contextlib
import inspect
import io
import os.path
import re
from . import compression
from . import transport
def tweak_parse_uri_docstring(f):
    buf = io.StringIO()
    seen = set()
    schemes = []
    examples = []
    for scheme, submodule in sorted(transport._REGISTRY.items()):
        if scheme == transport.NO_SCHEME or submodule in seen:
            continue
        schemes.append(scheme)
        seen.add(submodule)
        try:
            examples.extend(submodule.URI_EXAMPLES)
        except AttributeError:
            pass
    with contextlib.redirect_stdout(buf):
        print('    Supported URI schemes are:')
        print()
        for scheme in schemes:
            print('    * %s' % scheme)
        print()
        print('    Valid URI examples::')
        print()
        for example in examples:
            print('    * %s' % example)
    if f.__doc__:
        f.__doc__ = f.__doc__.replace(PLACEHOLDER, buf.getvalue())