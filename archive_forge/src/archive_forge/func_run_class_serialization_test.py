import json
import shutil
import tempfile
import unittest
import numpy as np
import tree
from keras.src import backend
from keras.src import ops
from keras.src import utils
from keras.src.backend.common import is_float_dtype
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.global_state import clear_session
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.models import Model
from keras.src.utils import traceback_utils
from keras.src.utils.shape_utils import map_shape_structure
def run_class_serialization_test(self, instance, custom_objects=None):
    from keras.src.saving import custom_object_scope
    from keras.src.saving import deserialize_keras_object
    from keras.src.saving import serialize_keras_object
    cls = instance.__class__
    config = instance.get_config()
    config_json = json.dumps(config, sort_keys=True, indent=4)
    ref_dir = dir(instance)[:]
    with custom_object_scope(custom_objects):
        revived_instance = cls.from_config(config)
    revived_config = revived_instance.get_config()
    revived_config_json = json.dumps(revived_config, sort_keys=True, indent=4)
    self.assertEqual(config_json, revived_config_json)
    self.assertEqual(set(ref_dir), set(dir(revived_instance)))
    serialized = serialize_keras_object(instance)
    serialized_json = json.dumps(serialized, sort_keys=True, indent=4)
    with custom_object_scope(custom_objects):
        revived_instance = deserialize_keras_object(json.loads(serialized_json))
    revived_config = revived_instance.get_config()
    revived_config_json = json.dumps(revived_config, sort_keys=True, indent=4)
    self.assertEqual(config_json, revived_config_json)
    new_dir = dir(revived_instance)[:]
    for lst in [ref_dir, new_dir]:
        if '__annotations__' in lst:
            lst.remove('__annotations__')
    self.assertEqual(set(ref_dir), set(new_dir))
    return revived_instance