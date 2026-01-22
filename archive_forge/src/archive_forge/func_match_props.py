from ._base import *
from .types import CreateSchemaType, UpdateSchemaType
from .models import LazyHasher, LazyUserSchema, LazyDBConfig, LazyDBSaveMetrics
@timed_cache(15)
def match_props(self, item, name, val, *args, **kwargs):
    item_data = jsonable_encoder(item)
    if not item_data.get(name):
        return False
    by_type = kwargs.get('ByType', False)
    if by_type:
        return isinstance(item_data[name], val)
    if isinstance(val, list) and isinstance(item_data[name], str):
        return bool(item_data[name] in val)
    if isinstance(val, str) and isinstance(item_data[name], list):
        return bool(val in item_data[name])
    if isinstance(val, str) and val == 'NotNone':
        return bool(item_data[name] is not None)
    if isinstance(val, str) and val == 'NotNoneType':
        return isinstance(item_data[name], NoneType)
    return item_data[name] == val