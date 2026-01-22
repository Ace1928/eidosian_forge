from __future__ import annotations
import binascii
import collections
import datetime
import enum
import glob
import io
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import numpy
from typing import TYPE_CHECKING, BinaryIO, cast, final, overload
def write_fsspec(self, jsonfile: str | os.PathLike[Any] | TextIO, /, url: str, *, quote: bool | None=None, groupname: str | None=None, templatename: str | None=None, codec_id: str | None=None, version: int | None=None, _append: bool=False, _close: bool=True) -> None:
    """Write fsspec ReferenceFileSystem as JSON to file.

        Parameters:
            jsonfile:
                Name or open file handle of output JSON file.
            url:
                Remote location of TIFF file(s) without file name(s).
            quote:
                Quote file names, that is, replace ' ' with '%20'.
                The default is True.
            groupname:
                Zarr group name.
            templatename:
                Version 1 URL template name. The default is 'u'.
            codec_id:
                Name of Numcodecs codec to decode files or chunks.
            version:
                Version of fsspec file to write. The default is 0.
            _append, _close:
                Experimental API.

        References:
            - `fsspec ReferenceFileSystem format
              <https://github.com/fsspec/kerchunk>`_

        """
    from urllib.parse import quote as quote_
    kwargs = self._kwargs.copy()
    if codec_id is not None:
        pass
    elif self._imread == imread:
        codec_id = 'tifffile'
    elif 'imagecodecs.' in self._imread.__module__:
        if self._imread.__name__ != 'imread' or 'codec' not in self._kwargs:
            raise ValueError('cannot determine codec_id')
        codec = kwargs.pop('codec')
        if isinstance(codec, (list, tuple)):
            codec = codec[0]
        if callable(codec):
            codec = codec.__name__.split('_')[0]
        codec_id = {'apng': 'imagecodecs_apng', 'avif': 'imagecodecs_avif', 'gif': 'imagecodecs_gif', 'heif': 'imagecodecs_heif', 'jpeg': 'imagecodecs_jpeg', 'jpeg8': 'imagecodecs_jpeg', 'jpeg12': 'imagecodecs_jpeg', 'jpeg2k': 'imagecodecs_jpeg2k', 'jpegls': 'imagecodecs_jpegls', 'jpegxl': 'imagecodecs_jpegxl', 'jpegxr': 'imagecodecs_jpegxr', 'ljpeg': 'imagecodecs_ljpeg', 'lerc': 'imagecodecs_lerc', 'png': 'imagecodecs_png', 'qoi': 'imagecodecs_qoi', 'tiff': 'imagecodecs_tiff', 'webp': 'imagecodecs_webp', 'zfp': 'imagecodecs_zfp'}[codec]
    else:
        raise ValueError('cannot determine codec_id')
    if url is None:
        url = ''
    elif url and url[-1] != '/':
        url += '/'
    if groupname is None:
        groupname = ''
    elif groupname and groupname[-1] != '/':
        groupname += '/'
    refs: dict[str, Any] = {}
    if version == 1:
        if _append:
            raise ValueError('cannot append to version 1 files')
        if templatename is None:
            templatename = 'u'
        refs['version'] = 1
        refs['templates'] = {templatename: url}
        refs['gen'] = []
        refs['refs'] = refzarr = {}
        url = '{{%s}}' % templatename
    else:
        refzarr = refs
    if groupname and (not _append):
        refzarr['.zgroup'] = ZarrStore._json({'zarr_format': 2}).decode()
    for key, value in self._store.items():
        if '.zarray' in key:
            value = json.loads(value)
            value['compressor'] = {'id': codec_id, **kwargs}
            value = ZarrStore._json(value)
        refzarr[groupname + key] = value.decode()
    fh: TextIO
    if hasattr(jsonfile, 'write'):
        fh = jsonfile
    else:
        fh = open(jsonfile, 'w', encoding='utf-8')
    if version == 1:
        fh.write(json.dumps(refs, indent=1).rsplit('}"', 1)[0] + '}"')
        indent = '  '
    elif _append:
        fh.write(',\n')
        fh.write(json.dumps(refs, indent=1)[2:-2])
        indent = ' '
    else:
        fh.write(json.dumps(refs, indent=1)[:-2])
        indent = ' '
    prefix = len(self._commonpath)
    for key, value in self._store.items():
        if '.zarray' in key:
            value = json.loads(value)
            for index, filename in sorted(self._lookup.items(), key=lambda x: x[0]):
                filename = filename[prefix:].replace('\\', '/')
                if quote is None or quote:
                    filename = quote_(filename)
                if filename[0] == '/':
                    filename = filename[1:]
                indexstr = '.'.join((str(i) for i in index))
                fh.write(f',\n{indent}"{groupname}{indexstr}": ["{url}{filename}"]')
    if version == 1:
        fh.write('\n }\n}')
    elif _close:
        fh.write('\n}')
    if not hasattr(jsonfile, 'write'):
        fh.close()