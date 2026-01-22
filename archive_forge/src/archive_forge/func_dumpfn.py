from __future__ import annotations
import json
import os
from typing import TYPE_CHECKING
from monty.io import zopen
from monty.json import MontyDecoder, MontyEncoder
from monty.msgpack import default, object_hook
def dumpfn(obj: object, fn: Union[str, Path], *args, fmt=None, **kwargs) -> None:
    """
    Dump to a json/yaml directly by filename instead of a
    File-like object. File may also be a BZ2 (".BZ2") or GZIP (".GZ", ".Z")
    compressed file.
    For YAML, ruamel.yaml must be installed. The file type is automatically
    detected from the file extension (case insensitive). YAML is assumed if the
    filename contains ".yaml" or ".yml".
    Msgpack is assumed if the filename contains ".mpk".
    JSON is otherwise assumed.

    Args:
        obj (object): Object to dump.
        fn (str/Path): filename or pathlib.Path.
        *args: Any of the args supported by json/yaml.dump.
        **kwargs: Any of the kwargs supported by json/yaml.dump.

    Returns:
        (object) Result of json.load.
    """
    if fmt is None:
        basename = os.path.basename(fn).lower()
        if '.mpk' in basename:
            fmt = 'mpk'
        elif any((ext in basename for ext in ('.yaml', '.yml'))):
            fmt = 'yaml'
        else:
            fmt = 'json'
    if fmt == 'mpk':
        if msgpack is None:
            raise RuntimeError('Loading of message pack files is not possible as msgpack-python is not installed.')
        if 'default' not in kwargs:
            kwargs['default'] = default
        with zopen(fn, 'wb') as fp:
            msgpack.dump(obj, fp, *args, **kwargs)
    else:
        with zopen(fn, 'wt') as fp:
            if fmt == 'yaml':
                if YAML is None:
                    raise RuntimeError('Loading of YAML files requires ruamel.yaml.')
                yaml = YAML()
                yaml.dump(obj, fp, *args, **kwargs)
            elif fmt == 'json':
                if 'cls' not in kwargs:
                    kwargs['cls'] = MontyEncoder
                fp.write(json.dumps(obj, *args, **kwargs))
            else:
                raise TypeError(f'Invalid format: {fmt}')