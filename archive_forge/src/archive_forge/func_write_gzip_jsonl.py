from typing import Union, Iterable, Sequence, Any, Optional, Iterator
import sys
import json as _builtin_json
import gzip
from . import ujson
from .util import force_path, force_string, FilePath, JSONInput, JSONOutput
def write_gzip_jsonl(path: FilePath, lines: Iterable[JSONInput], append: bool=False, append_new_line: bool=True) -> None:
    """Create a .jsonl.gz file and dump contents.

    location (FilePath): The file path.
    lines (Sequence[JSONInput]): The JSON-serializable contents of each line.
    append (bool): Whether or not to append to the location. Appending to .gz files is generally not recommended, as it
        doesn't allow the algorithm to take advantage of all data when compressing - files may hence be poorly
        compressed.
    append_new_line (bool): Whether or not to write a new line before appending
        to the file.
    """
    mode = 'a' if append else 'w'
    file_path = force_path(path, require_exists=False)
    with gzip.open(file_path, mode=mode) as f:
        if append and append_new_line:
            f.write('\n'.encode('utf-8'))
        f.writelines([(json_dumps(line) + '\n').encode('utf-8') for line in lines])