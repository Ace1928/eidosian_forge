from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
def set_style_texts(edits: typing.List[StyleDeclarationEdit], node_for_property_syntax_validation: typing.Optional[dom.NodeId]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[CSSStyle]]:
    """
    Applies specified style edits one after another in the given order.

    :param edits:
    :param node_for_property_syntax_validation: **(EXPERIMENTAL)** *(Optional)* NodeId for the DOM node in whose context custom property declarations for registered properties should be validated. If omitted, declarations in the new rule text can only be validated statically, which may produce incorrect results if the declaration contains a var() for example.
    :returns: The resulting styles after modification.
    """
    params: T_JSON_DICT = dict()
    params['edits'] = [i.to_json() for i in edits]
    if node_for_property_syntax_validation is not None:
        params['nodeForPropertySyntaxValidation'] = node_for_property_syntax_validation.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'CSS.setStyleTexts', 'params': params}
    json = (yield cmd_dict)
    return [CSSStyle.from_json(i) for i in json['styles']]