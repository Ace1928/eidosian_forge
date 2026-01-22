from typing import Union, Iterable, Sequence, Any, Optional, Iterator
import sys
import json as _builtin_json
import gzip
from . import ujson
from .util import force_path, force_string, FilePath, JSONInput, JSONOutput
def write_jsonl(path: FilePath, lines: Iterable[JSONInput], append: bool=False, append_new_line: bool=True) -> None:
    """Create a .jsonl file and dump contents or write to standard output.

    location (FilePath): The file path. "-" for writing to stdout.
    lines (Sequence[JSONInput]): The JSON-serializable contents of each line.
    append (bool): Whether or not to append to the location.
    append_new_line (bool): Whether or not to write a new line before appending
        to the file.
    """
    if path == '-':
        for line in lines:
            print(json_dumps(line))
    else:
        mode = 'a' if append else 'w'
        file_path = force_path(path, require_exists=False)
        with file_path.open(mode, encoding='utf-8') as f:
            if append and append_new_line:
                f.write('\n')
            for line in lines:
                f.write(json_dumps(line) + '\n')