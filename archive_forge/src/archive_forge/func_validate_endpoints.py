import requests
import aiohttp
import simdjson as json
from lazyops.utils import timed_cache
from lazyops.serializers import async_cache
from ._base import lazyclass, dataclass, List, Union, Any, Dict
from .tfserving_pb2 import TFSModelConfig, TFSConfig
def validate_endpoints(self):
    for n, version in enumerate(self.config.model_versions):
        r = self.get_metadata(label=version.label)
        if r.get('error'):
            self.config.model_versions[n].label = None
        r = self.get_metadata(ver=str(version.step))
        if r.get('error'):
            self.config.model_versions[n].step = None