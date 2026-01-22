import itertools
import re
import warnings
from ..api import APIClient
from ..constants import DEFAULT_DATA_CHUNK_SIZE
from ..errors import BuildError, ImageLoadError, InvalidArgument
from ..utils import parse_repository_tag
from ..utils.json_stream import json_stream
from .resource import Collection, Model
def normalize_platform(platform, engine_info):
    if platform is None:
        platform = {}
    if 'os' not in platform:
        platform['os'] = engine_info['Os']
    if 'architecture' not in platform:
        platform['architecture'] = engine_info['Arch']
    return platform