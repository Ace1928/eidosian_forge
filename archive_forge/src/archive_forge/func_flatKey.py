from collections import defaultdict
from string import Formatter
from typing import Any, Dict, Optional
from ._interfaces import LogEvent
def flatKey(self, fieldName: str, formatSpec: Optional[str], conversion: Optional[str]) -> str:
    """
        Compute a string key for a given field/format/conversion.

        @param fieldName: A format field name.
        @param formatSpec: A format spec.
        @param conversion: A format field conversion type.

        @return: A key specific to the given field, format and conversion, as
            well as the occurrence of that combination within this
            L{KeyFlattener}'s lifetime.
        """
    if formatSpec is None:
        formatSpec = ''
    if conversion is None:
        conversion = ''
    result = '{fieldName}!{conversion}:{formatSpec}'.format(fieldName=fieldName, formatSpec=formatSpec, conversion=conversion)
    self.keys[result] += 1
    n = self.keys[result]
    if n != 1:
        result += '/' + str(self.keys[result])
    return result