import contextlib
from dataclasses import dataclass
import logging
import os
import ray
from ray import cloudpickle
from ray.types import ObjectRef
from ray.workflow import common, workflow_storage
from typing import Any, Dict, Generator, List, Optional, Tuple, TYPE_CHECKING
from collections import ChainMap
import io
def dump_to_storage(key: str, obj: Any, workflow_id: str, storage: 'workflow_storage.WorkflowStorage', update_existing=True) -> None:
    """Serializes and puts arbitrary object, handling references. The object will
        be uploaded at `paths`. Any object references will be uploaded to their
        global, remote storage.

    Args:
        key: The key of the object.
        obj: The object to serialize. If it contains object references, those
                will be serialized too.
        workflow_id: The workflow id.
        storage: The storage to use. If obj contains object references,
                `storage.put` will be called on them individually.
        update_existing: If False, the object will not be uploaded if the path
                exists.
    """
    if not update_existing:
        if storage._exists(key):
            return
    tasks = []

    class ObjectRefPickler(cloudpickle.CloudPickler):
        _object_ref_reducer = {ray.ObjectRef: lambda ref: _reduce_objectref(workflow_id, ref, tasks)}
        dispatch_table = ChainMap(_object_ref_reducer, cloudpickle.CloudPickler.dispatch_table)
        dispatch = dispatch_table
    ray.get(tasks)
    with io.BytesIO() as f:
        pickler = ObjectRefPickler(f)
        pickler.dump(obj)
        f.seek(0)
        storage._storage.put(key, f.read())