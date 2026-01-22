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
def fail_callback_output(output_value, output):
    valid_children = (str, int, float, type(None), Component)
    valid_props = (str, int, float, type(None), tuple, MutableSequence)

    def _raise_invalid(bad_val, outer_val, path, index=None, toplevel=False):
        bad_type = type(bad_val).__name__
        outer_id = f'(id={outer_val.id:s})' if getattr(outer_val, 'id', False) else ''
        outer_type = type(outer_val).__name__
        if toplevel:
            location = dedent('\n                The value in question is either the only value returned,\n                or is in the top level of the returned list,\n                ')
        else:
            index_string = '[*]' if index is None else f'[{index:d}]'
            location = dedent(f'\n                The value in question is located at\n                {index_string} {outer_type} {outer_id}\n                {path},\n                ')
        obj = 'tree with one value' if not toplevel else 'value'
        raise exceptions.InvalidCallbackReturnValue(dedent(f'\n                The callback for `{output!r}`\n                returned a {obj:s} having type `{bad_type}`\n                which is not JSON serializable.\n\n                {location}\n                and has string representation\n                `{bad_val}`\n\n                In general, Dash properties can only be\n                dash components, strings, dictionaries, numbers, None,\n                or lists of those.\n                '))

    def _valid_child(val):
        return isinstance(val, valid_children)

    def _valid_prop(val):
        return isinstance(val, valid_props)

    def _can_serialize(val):
        if not (_valid_child(val) or _valid_prop(val)):
            return False
        try:
            to_json(val)
        except TypeError:
            return False
        return True

    def _validate_value(val, index=None):
        if isinstance(val, Component):
            unserializable_items = []
            for p, j in val._traverse_with_paths():
                if not _valid_child(j):
                    _raise_invalid(bad_val=j, outer_val=val, path=p, index=index)
                if not _can_serialize(j):
                    unserializable_items = [i for i in unserializable_items if not p.startswith(i[0])]
                    if unserializable_items:
                        break
                    if all((not i[0].startswith(p) for i in unserializable_items)):
                        unserializable_items.append((p, j))
                child = getattr(j, 'children', None)
                if not isinstance(child, (tuple, MutableSequence)):
                    if child and (not _can_serialize(child)):
                        _raise_invalid(bad_val=child, outer_val=val, path=p + '\n' + '[*] ' + type(child).__name__, index=index)
            if unserializable_items:
                p, j = unserializable_items[0]
                _raise_invalid(bad_val=j, outer_val=val, path=p, index=index)
            child = getattr(val, 'children', None)
            if not isinstance(child, (tuple, MutableSequence)):
                if child and (not _can_serialize(val)):
                    _raise_invalid(bad_val=child, outer_val=val, path=type(child).__name__, index=index)
        if not _can_serialize(val):
            _raise_invalid(bad_val=val, outer_val=type(val).__name__, path='', index=index, toplevel=True)
    if isinstance(output_value, list):
        for i, val in enumerate(output_value):
            _validate_value(val, index=i)
    else:
        _validate_value(output_value)
    raise exceptions.InvalidCallbackReturnValue(f'\n        The callback for output `{output!r}`\n        returned a value which is not JSON serializable.\n\n        In general, Dash properties can only be dash components, strings,\n        dictionaries, numbers, None, or lists of those.\n        ')