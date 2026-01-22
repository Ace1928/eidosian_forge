import json
from typing import Any, List
from urllib import parse
import pathlib
from filelock import FileLock
from ray.workflow.storage.base import Storage
from ray.workflow.storage.filesystem import FilesystemStorageImpl
import ray.cloudpickle
from ray.workflow import serialization_context
def update_count(self, op: str, key):
    counter = None
    with open(self._op_counter, 'rb') as f:
        counter = ray.cloudpickle.load(f)
    if op not in counter:
        counter[op] = []
    counter[op].append(key)
    with open(self._op_counter, 'wb') as f:
        ray.cloudpickle.dump(counter, f)