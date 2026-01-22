from __future__ import annotations
from typing import Any, Dict, Optional, Union, Type
from lazyops.utils.lazy import lazy_import
from .base import BinaryBaseSerializer, BaseModel, SchemaType, ObjectValue, logger
from ._json import default_json
def default_serialization_hook(self, obj: ObjectValue):
    """
        Default Serialization Hook
        """
    if not isinstance(obj, BaseModel) and (not hasattr(obj, 'model_dump')):
        logger.info(f'Invalid Object Type: |r|{type(obj)}|e| {obj}', colored=True, prefix='msgpack')
        return obj
    if self.disable_object_serialization:
        return obj.model_dump_json(**self.serialization_obj_kwargs)
    obj_class_name = self.fetch_object_classname(obj)
    if obj_class_name not in self.serialization_schemas:
        self.serialization_schemas[obj_class_name] = obj.__class__
    data = obj.model_dump(mode='json', **self.serialization_obj_kwargs)
    data['__class__'] = obj_class_name
    return msgpack.ExtType(2, self.jsonlib.dumps(data).encode(self.encoding))