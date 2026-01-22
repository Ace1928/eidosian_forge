import datetime
import math
import typing as t
from wandb.util import (
@staticmethod
def type_of(py_obj: t.Optional[t.Any]) -> 'Type':
    if py_obj.__class__ == float and math.isnan(py_obj):
        return NoneType()
    if _is_artifact_string(py_obj) or _is_artifact_version_weave_dict(py_obj):
        return TypeRegistry.types_by_name().get('artifactVersion')()
    class_handler = TypeRegistry.types_by_class().get(py_obj.__class__)
    _type = None
    if class_handler:
        _type = class_handler.from_obj(py_obj)
    else:
        _type = PythonObjectType.from_obj(py_obj)
    return _type