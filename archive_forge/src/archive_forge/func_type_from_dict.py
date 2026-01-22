import datetime
import math
import typing as t
from wandb.util import (
@staticmethod
def type_from_dict(json_dict: t.Dict[str, t.Any], artifact: t.Optional['Artifact']=None) -> 'Type':
    wb_type = json_dict.get('wb_type')
    if wb_type is None:
        TypeError('json_dict must contain `wb_type` key')
    _type = TypeRegistry.types_by_name().get(wb_type)
    if _type is None:
        TypeError(f'missing type handler for {wb_type}')
    return _type.from_json(json_dict, artifact)