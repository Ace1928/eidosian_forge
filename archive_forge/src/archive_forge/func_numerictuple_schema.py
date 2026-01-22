import json
import textwrap
@classmethod
def numerictuple_schema(cls, p, safe=False):
    schema = cls.tuple_schema(p, safe=safe)
    schema['additionalItems'] = {'type': 'number'}
    return schema