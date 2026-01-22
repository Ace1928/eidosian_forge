import json
import textwrap
@classmethod
def number_schema(cls, p, safe=False):
    schema = {'type': p.__class__.__name__.lower()}
    return cls.declare_numeric_bounds(schema, p.bounds, p.inclusive_bounds)