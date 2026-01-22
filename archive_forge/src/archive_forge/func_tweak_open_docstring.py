import contextlib
import inspect
import io
import os.path
import re
from . import compression
from . import transport
def tweak_open_docstring(f):
    buf = io.StringIO()
    seen = set()
    root_path = os.path.dirname(os.path.dirname(__file__))
    with contextlib.redirect_stdout(buf):
        print('    smart_open supports the following transport mechanisms:')
        print()
        for scheme, submodule in sorted(transport._REGISTRY.items()):
            if scheme == transport.NO_SCHEME or submodule in seen:
                continue
            seen.add(submodule)
            relpath = os.path.relpath(submodule.__file__, start=root_path)
            heading = '%s (%s)' % (scheme, relpath)
            print('    %s' % heading)
            print('    %s' % ('~' * len(heading)))
            print('    %s' % submodule.__doc__.split('\n')[0])
            print()
            kwargs = extract_kwargs(submodule.open.__doc__)
            if kwargs:
                print(to_docstring(kwargs, lpad=u'    '))
        print('    Examples')
        print('    --------')
        print()
        print(extract_examples_from_readme_rst())
        print('    This function also supports transparent compression and decompression ')
        print('    using the following codecs:')
        print()
        for extension in compression.get_supported_extensions():
            print('    * %s' % extension)
        print()
        print('    The function depends on the file extension to determine the appropriate codec.')
    if f.__doc__:
        f.__doc__ = f.__doc__.replace(PLACEHOLDER, buf.getvalue())