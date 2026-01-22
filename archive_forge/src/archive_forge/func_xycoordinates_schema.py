import json
import textwrap
@classmethod
def xycoordinates_schema(cls, p, safe=False):
    return cls.numerictuple_schema(p, safe=safe)