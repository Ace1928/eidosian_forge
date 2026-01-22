import warnings
from typing import Any, Callable
from sphinx.deprecation import RemovedInSphinx60Warning
def execfile_(filepath: str, _globals: Any, open: Callable=open) -> None:
    warnings.warn('execfile_() is deprecated', RemovedInSphinx60Warning, stacklevel=2)
    from sphinx.util.osutil import fs_encoding
    with open(filepath, 'rb') as f:
        source = f.read()
    filepath_enc = filepath.encode(fs_encoding)
    code = compile(source, filepath_enc, 'exec')
    exec(code, _globals)