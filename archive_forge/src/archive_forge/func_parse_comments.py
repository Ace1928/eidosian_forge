from datetime import datetime
import logging
import os
from typing import (
import warnings
import numpy as np
from ..core.request import Request, IOMode, InitializationError
from ..core.v3_plugin_api import PluginV3, ImageProperties
@staticmethod
def parse_comments(comments: Sequence[str], version: Tuple[int, int]) -> Dict[str, Any]:
    """Extract SDT-control metadata from comments

        Parameters
        ----------
        comments
            List of SPE file comments, typically ``metadata["comments"]``.
        version
            Major and minor version of SDT-control metadata format

        Returns
        -------
        Dict of metadata
        """
    sdt_md = {}
    for minor in range(version[1] + 1):
        try:
            cmt = __class__.comment_fields[version[0], minor]
        except KeyError:
            continue
        for name, spec in cmt.items():
            try:
                v = spec.cvt(comments[spec.n][spec.slice])
                if spec.scale is not None:
                    v *= spec.scale
                sdt_md[name] = v
            except Exception as e:
                warnings.warn(f'Failed to decode SDT-control metadata field `{name}`: {e}')
                sdt_md[name] = None
    if version not in __class__.comment_fields:
        supported_ver = ', '.join(map(lambda x: f'{x[0]}.{x[1]:02}', __class__.comment_fields))
        warnings.warn(f'Unsupported SDT-control metadata version {version[0]}.{version[1]:02}. Only versions {supported_ver} are supported. Some or all SDT-control metadata may be missing.')
    comment = comments[0] + comments[2]
    sdt_md['comment'] = comment.strip()
    return sdt_md