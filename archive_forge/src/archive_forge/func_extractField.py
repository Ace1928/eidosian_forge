from collections import defaultdict
from string import Formatter
from typing import Any, Dict, Optional
from ._interfaces import LogEvent
def extractField(field: str, event: LogEvent) -> Any:
    """
    Extract a given format field from the given event.

    @param field: A string describing a format field or log key.  This is the
        text that would normally fall between a pair of curly braces in a
        format string: for example, C{"key[2].attribute"}.  If a conversion is
        specified (the thing after the C{"!"} character in a format field) then
        the result will always be str.
    @param event: A log event.

    @return: A value extracted from the field.

    @raise KeyError: if the field is not found in the given event.
    """
    keyFlattener = KeyFlattener()
    [[literalText, fieldName, formatSpec, conversion]] = aFormatter.parse('{' + field + '}')
    assert fieldName is not None
    key = keyFlattener.flatKey(fieldName, formatSpec, conversion)
    if 'log_flattened' not in event:
        flattenEvent(event)
    return event['log_flattened'][key]