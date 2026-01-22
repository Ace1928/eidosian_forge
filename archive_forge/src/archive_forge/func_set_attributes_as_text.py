from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def set_attributes_as_text(node_id: NodeId, text: str, name: typing.Optional[str]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Sets attributes on element with given id. This method is useful when user edits some existing
    attribute value and types in several attribute name/value pairs.

    :param node_id: Id of the element to set attributes for.
    :param text: Text with a number of attributes. Will parse this text using HTML parser.
    :param name: *(Optional)* Attribute name to replace with new attributes derived from text in case text parsed successfully.
    """
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    params['text'] = text
    if name is not None:
        params['name'] = name
    cmd_dict: T_JSON_DICT = {'method': 'DOM.setAttributesAsText', 'params': params}
    json = (yield cmd_dict)