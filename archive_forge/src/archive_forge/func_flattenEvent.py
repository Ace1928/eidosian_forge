from collections import defaultdict
from string import Formatter
from typing import Any, Dict, Optional
from ._interfaces import LogEvent
def flattenEvent(event: LogEvent) -> None:
    """
    Flatten the given event by pre-associating format fields with specific
    objects and callable results in a L{dict} put into the C{"log_flattened"}
    key in the event.

    @param event: A logging event.
    """
    if event.get('log_format', None) is None:
        return
    if 'log_flattened' in event:
        fields = event['log_flattened']
    else:
        fields = {}
    keyFlattener = KeyFlattener()
    for literalText, fieldName, formatSpec, conversion in aFormatter.parse(event['log_format']):
        if fieldName is None:
            continue
        if conversion != 'r':
            conversion = 's'
        flattenedKey = keyFlattener.flatKey(fieldName, formatSpec, conversion)
        structuredKey = keyFlattener.flatKey(fieldName, formatSpec, '')
        if flattenedKey in fields:
            continue
        if fieldName.endswith('()'):
            fieldName = fieldName[:-2]
            callit = True
        else:
            callit = False
        field = aFormatter.get_field(fieldName, (), event)
        fieldValue = field[0]
        if conversion == 'r':
            conversionFunction = repr
        else:
            conversionFunction = str
        if callit:
            fieldValue = fieldValue()
        flattenedValue = conversionFunction(fieldValue)
        fields[flattenedKey] = flattenedValue
        fields[structuredKey] = fieldValue
    if fields:
        event['log_flattened'] = fields