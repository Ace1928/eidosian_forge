import requests
import aiohttp
import simdjson as json
from lazyops.utils import timed_cache
from lazyops.serializers import async_cache
from ._base import lazyclass, dataclass, List, Union, Any, Dict
from .tfserving_pb2 import TFSModelConfig, TFSConfig
@classmethod
def from_config_file(cls, url, filepath: str, ver: str='v1', **kwargs):
    configs = TFSConfig.from_config_file(filepath, as_obj=True)
    return TFServeModel(url=url, configs=configs, ver=ver, **kwargs)