import dis
import hashlib
import importlib
import inspect
import json
import logging
import os
import threading
import time
import traceback
from collections import defaultdict, namedtuple
from typing import Optional, Callable
import ray
import ray._private.profiling as profiling
from ray import cloudpickle as pickle
from ray._private import ray_constants
from ray._private.inspect_util import (
from ray._private.ray_constants import KV_NAMESPACE_FUNCTION_TABLE
from ray._private.utils import (
from ray._private.serialization import pickle_dumps
from ray._raylet import (
def export_actor_class(self, Class, actor_creation_function_descriptor, actor_method_names):
    if self._worker.load_code_from_local:
        module_name, class_name = (actor_creation_function_descriptor.module_name, actor_creation_function_descriptor.class_name)
        if self.load_function_or_class_from_local(module_name, class_name) is not None:
            return
    assert not self._worker.current_job_id.is_nil(), 'You might have started a background thread in a non-actor task, please make sure the thread finishes before the task finishes.'
    job_id = self._worker.current_job_id
    key = make_function_table_key(b'ActorClass', job_id, actor_creation_function_descriptor.function_id.binary())
    serialized_actor_class = pickle_dumps(Class, f'Could not serialize the actor class {actor_creation_function_descriptor.repr}')
    actor_class_info = {'class_name': actor_creation_function_descriptor.class_name.split('.')[-1], 'module': actor_creation_function_descriptor.module_name, 'class': serialized_actor_class, 'job_id': job_id.binary(), 'collision_identifier': self.compute_collision_identifier(Class), 'actor_method_names': json.dumps(list(actor_method_names))}
    check_oversized_function(actor_class_info['class'], actor_class_info['class_name'], 'actor', self._worker)
    self._worker.gcs_client.internal_kv_put(key, pickle.dumps(actor_class_info), True, KV_NAMESPACE_FUNCTION_TABLE)