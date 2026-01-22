import requests
import aiohttp
import simdjson as json
from lazyops.utils import timed_cache
from lazyops.serializers import async_cache
from ._base import lazyclass, dataclass, List, Union, Any, Dict
from .tfserving_pb2 import TFSModelConfig, TFSConfig
@property
def default_url(self):
    if self.config.default_label:
        return self.default_endpoint + f'/labels/{self.config.default_label}'
    if self.config.default_version:
        return self.default_endpoint + f'/versions/{self.config.default_version}'
    return self.default_endpoint