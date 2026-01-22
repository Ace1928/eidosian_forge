import json
import textwrap
@classmethod
def classselector_schema(cls, p, safe=False):
    return cls.class__schema(p.class_, safe=safe)