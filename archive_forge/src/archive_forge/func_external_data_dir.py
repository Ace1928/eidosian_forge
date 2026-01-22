from __future__ import annotations # isort:skip
import hashlib
import json
from os.path import splitext
from pathlib import Path
from sys import stdout
from typing import TYPE_CHECKING, Any, TextIO
from urllib.parse import urljoin
from urllib.request import urlopen
def external_data_dir(*, create: bool=False) -> Path:
    try:
        import yaml
    except ImportError:
        raise RuntimeError("'yaml' and 'pyyaml' are required to use bokeh.sampledata functions")
    bokeh_dir = _bokeh_dir(create=create)
    data_dir = bokeh_dir / 'data'
    try:
        config = yaml.safe_load(open(bokeh_dir / 'config'))
        data_dir = Path.expanduser(config['sampledata_dir'])
    except (OSError, TypeError):
        pass
    if not data_dir.exists():
        if not create:
            raise RuntimeError('bokeh sample data directory does not exist, please execute bokeh.sampledata.download()')
        print(f'Creating {data_dir} directory')
        try:
            data_dir.mkdir()
        except OSError:
            raise RuntimeError(f'could not create bokeh data directory at {data_dir}')
    elif not data_dir.is_dir():
        raise RuntimeError(f'{data_dir} exists but is not a directory')
    return data_dir