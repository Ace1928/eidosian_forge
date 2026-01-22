from __future__ import annotations
import json
from typing import Any, Dict, Optional, Union, Type
from lazyops.utils.lazy import lazy_import
from .base import BaseSerializer, ObjectValue, SchemaType, BaseModel, logger
def encode_one(self, value: Union[Any, SchemaType], **kwargs) -> str:
    """
        Encode the value with the JSON Library
        """
    try:
        if hasattr(value, 'model_dump'):
            if not self.disable_object_serialization:
                obj_class_name = self.fetch_object_classname(value)
                if obj_class_name not in self.serialization_schemas:
                    self.serialization_schemas[obj_class_name] = value.__class__
            value = value.model_dump(mode='json', round_trip=True, **self.serialization_obj_kwargs)
            if not self.disable_object_serialization:
                value['__class__'] = obj_class_name
    except Exception as e:
        logger.info(f'Error Encoding Value: |r|({type(value)}) {e}|e| {str(value)[:1000]}', colored=True)
    try:
        return self.jsonlib.dumps(value, **kwargs)
    except Exception as e:
        logger.info(f'Error Encoding Value: |r|({type(value)}) {e}|e| {str(value)[:1000]}', colored=True, prefix=self.jsonlib_name)
        if self.raise_errors:
            raise e
    return None