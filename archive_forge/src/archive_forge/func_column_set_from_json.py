import re
import sys
import ovs.db.parser
import ovs.db.types
from ovs.db import error
def column_set_from_json(json, columns):
    if json is None:
        return tuple(columns)
    elif not isinstance(json, list):
        raise error.Error('array of distinct column names expected', json)
    else:
        for column_name in json:
            if not isinstance(column_name, str):
                raise error.Error('array of distinct column names expected', json)
            elif column_name not in columns:
                raise error.Error('%s is not a valid column name' % column_name, json)
        if len(set(json)) != len(json):
            raise error.Error('array of distinct column names expected', json)
        return tuple([columns[column_name] for column_name in json])