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
def detect_series(pages: TiffPages | list[TiffPage | TiffFrame | None], series: list[TiffPageSeries], /) -> list[TiffPageSeries] | None:
    shape: tuple[int, ...] | None
    reshape: tuple[int, ...]
    page: TiffPage | TiffFrame | None
    keyframe: TiffPage
    subifds: list[TiffPage | TiffFrame | None] = []
    subifd: TiffPage | TiffFrame
    keysubifd: TiffPage
    axes: str | None
    name: str
    lenpages = len(pages)
    index = 0
    while True:
        if index >= lenpages:
            break
        if isinstance(pages, TiffPages):
            pages.set_keyframe(index)
            keyframe = cast(TiffPage, pages.keyframe)
        else:
            keyframe = cast(TiffPage, pages[0])
        if keyframe.shaped_description is None:
            logger().error(f'{self!r} invalid shaped series metadata or corrupted file')
            return None
        axes = None
        shape = None
        metadata = shaped_description_metadata(keyframe.shaped_description)
        name = metadata.get('name', '')
        reshape = metadata['shape']
        truncated = None if keyframe.subifds is None else False
        truncated = metadata.get('truncated', truncated)
        if 'axes' in metadata:
            axes = cast(str, metadata['axes'])
            if len(axes) == len(reshape):
                shape = reshape
            else:
                axes = ''
                logger().error(f'{self!r} shaped series axes do not match shape')
        spages: list[TiffPage | TiffFrame | None] = [keyframe]
        size = product(reshape)
        if size > 0:
            npages, mod = divmod(size, product(keyframe.shape))
        else:
            npages = 1
            mod = 0
        if mod:
            logger().error(f'{self!r} shaped series shape does not match page shape')
            return None
        if 1 < npages <= lenpages - index:
            assert keyframe._dtype is not None
            size *= keyframe._dtype.itemsize
            if truncated:
                npages = 1
            else:
                page = pages[index + 1]
                if keyframe.is_final and page is not None and (keyframe.offset + size < page.offset) and (keyframe.subifds is None):
                    truncated = False
                else:
                    truncated = False
                    for j in range(index + 1, index + npages):
                        page = pages[j]
                        assert page is not None
                        page.keyframe = keyframe
                        spages.append(page)
        append(series, spages, axes, shape, reshape, name, truncated)
        index += npages
        if keyframe.subifds:
            subifds_size = len(keyframe.subifds)
            for i, offset in enumerate(keyframe.subifds):
                if offset < 8:
                    continue
                subifds = []
                for j, page in enumerate(spages):
                    try:
                        if page is None or page.subifds is None or len(page.subifds) < subifds_size:
                            raise ValueError(f'{page!r} contains invalid subifds')
                        self._fh.seek(page.subifds[i])
                        if j == 0:
                            subifd = TiffPage(self, (page.index, i))
                            keysubifd = subifd
                        else:
                            subifd = TiffFrame(self, (page.index, i), keyframe=keysubifd)
                    except Exception as exc:
                        logger().error(f'{self!r} shaped series raised {exc!r}')
                        return None
                    subifds.append(subifd)
                if subifds:
                    series_or_none = detect_series(subifds, series)
                    if series_or_none is None:
                        return None
                    series = series_or_none
    return series