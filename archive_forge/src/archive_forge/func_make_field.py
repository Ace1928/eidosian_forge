from pandas.core.common import (
def make_field(arr, dtype=None):
    dtype = dtype or arr.dtype
    if arr.name is None:
        name = 'values'
    else:
        name = arr.name
    field = {'name': name, 'type': as_json_table_type(dtype)}
    if is_categorical_dtype(arr):
        if hasattr(arr, 'categories'):
            cats = arr.categories
            ordered = arr.ordered
        else:
            cats = arr.cat.categories
            ordered = arr.cat.ordered
        field['constraints'] = {'enum': list(cats)}
        field['ordered'] = ordered
    elif is_datetime64tz_dtype(arr):
        if hasattr(arr, 'dt'):
            field['tz'] = arr.dt.tz.zone
        else:
            field['tz'] = arr.tz.zone
    return field