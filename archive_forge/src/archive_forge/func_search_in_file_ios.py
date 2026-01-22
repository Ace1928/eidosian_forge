import os
import re
from parso import python_bytes_to_unicode
from jedi.debug import dbg
from jedi.file_io import KnownContentFileIO, FolderIO
from jedi.inference.names import SubModuleName
from jedi.inference.imports import load_module_from_path
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.gradual.conversion import convert_names
def search_in_file_ios(inference_state, file_io_iterator, name, limit_reduction=1, complete=False):
    parse_limit = _PARSED_FILE_LIMIT / limit_reduction
    open_limit = _OPENED_FILE_LIMIT / limit_reduction
    file_io_count = 0
    parsed_file_count = 0
    regex = re.compile('\\b' + re.escape(name) + ('' if complete else '\\b'))
    for file_io in file_io_iterator:
        file_io_count += 1
        m = _check_fs(inference_state, file_io, regex)
        if m is not None:
            parsed_file_count += 1
            yield m
            if parsed_file_count >= parse_limit:
                dbg('Hit limit of parsed files: %s', parse_limit)
                break
        if file_io_count >= open_limit:
            dbg('Hit limit of opened files: %s', open_limit)
            break