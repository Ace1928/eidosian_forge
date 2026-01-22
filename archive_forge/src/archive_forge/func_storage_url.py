import json
from typing import Any, List
from urllib import parse
import pathlib
from filelock import FileLock
from ray.workflow.storage.base import Storage
from ray.workflow.storage.filesystem import FilesystemStorageImpl
import ray.cloudpickle
from ray.workflow import serialization_context
@property
def storage_url(self) -> str:
    store_url = parse.quote_plus(self._wrapped_storage.storage_url)
    parsed_url = parse.ParseResult(scheme='debug', path=str(pathlib.Path(self._path).absolute()), netloc='', params='', query=f'storage={store_url}', fragment='')
    return parse.urlunparse(parsed_url)