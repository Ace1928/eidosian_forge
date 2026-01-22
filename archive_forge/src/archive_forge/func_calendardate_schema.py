import json
import textwrap
@classmethod
def calendardate_schema(cls, p, safe=False):
    return {'type': 'string', 'format': 'date'}