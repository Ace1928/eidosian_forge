import sys
from collections.abc import MutableSequence
import re
from textwrap import dedent
from keyword import iskeyword
import flask
from ._grouping import grouping_len, map_grouping
from .development.base_component import Component
from . import exceptions
from ._utils import (
def validate_long_callbacks(callback_map):
    all_outputs = set()
    input_indexed = {}
    for callback in callback_map.values():
        out = coerce_to_list(callback['output'])
        all_outputs.update(out)
        for o in out:
            input_indexed.setdefault(o, set())
            input_indexed[o].update(coerce_to_list(callback['raw_inputs']))
    for callback in (x for x in callback_map.values() if x.get('long')):
        long_info = callback['long']
        progress = long_info.get('progress', [])
        running = long_info.get('running', [])
        long_inputs = coerce_to_list(callback['raw_inputs'])
        outputs = set([x[0] for x in running] + progress)
        circular = [x for x in set((k for k, v in input_indexed.items() if v.intersection(outputs))) if x in long_inputs]
        if circular:
            raise exceptions.LongCallbackError(f'Long callback circular error!\n{circular} is used as input for a long callback but also used as output from an input that is updated with progress or running argument.')