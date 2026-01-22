import os
import re
from parso import python_bytes_to_unicode
from jedi.debug import dbg
from jedi.file_io import KnownContentFileIO, FolderIO
from jedi.inference.names import SubModuleName
from jedi.inference.imports import load_module_from_path
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.gradual.conversion import convert_names
def gitignored_paths(folder_io, file_io):
    ignored_paths_abs = set()
    ignored_paths_rel = set()
    for l in file_io.read().splitlines():
        if not l or l.startswith(b'#') or l.startswith(b'!') or (b'*' in l):
            continue
        p = l.decode('utf-8', 'ignore').rstrip('/')
        if '/' in p:
            name = p.lstrip('/')
            ignored_paths_abs.add(os.path.join(folder_io.path, name))
        else:
            name = p
            ignored_paths_rel.add((folder_io.path, name))
    return (ignored_paths_abs, ignored_paths_rel)