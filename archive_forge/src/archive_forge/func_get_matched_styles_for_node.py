from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
def get_matched_styles_for_node(node_id: dom.NodeId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[typing.Optional[CSSStyle], typing.Optional[CSSStyle], typing.Optional[typing.List[RuleMatch]], typing.Optional[typing.List[PseudoElementMatches]], typing.Optional[typing.List[InheritedStyleEntry]], typing.Optional[typing.List[InheritedPseudoElementMatches]], typing.Optional[typing.List[CSSKeyframesRule]], typing.Optional[typing.List[CSSPositionFallbackRule]], typing.Optional[typing.List[CSSPropertyRule]], typing.Optional[typing.List[CSSPropertyRegistration]], typing.Optional[CSSFontPaletteValuesRule], typing.Optional[dom.NodeId]]]:
    """
    Returns requested styles for a DOM node identified by ``nodeId``.

    :param node_id:
    :returns: A tuple with the following items:

        0. **inlineStyle** - *(Optional)* Inline style for the specified DOM node.
        1. **attributesStyle** - *(Optional)* Attribute-defined element style (e.g. resulting from "width=20 height=100%").
        2. **matchedCSSRules** - *(Optional)* CSS rules matching this node, from all applicable stylesheets.
        3. **pseudoElements** - *(Optional)* Pseudo style matches for this node.
        4. **inherited** - *(Optional)* A chain of inherited styles (from the immediate node parent up to the DOM tree root).
        5. **inheritedPseudoElements** - *(Optional)* A chain of inherited pseudo element styles (from the immediate node parent up to the DOM tree root).
        6. **cssKeyframesRules** - *(Optional)* A list of CSS keyframed animations matching this node.
        7. **cssPositionFallbackRules** - *(Optional)* A list of CSS position fallbacks matching this node.
        8. **cssPropertyRules** - *(Optional)* A list of CSS at-property rules matching this node.
        9. **cssPropertyRegistrations** - *(Optional)* A list of CSS property registrations matching this node.
        10. **cssFontPaletteValuesRule** - *(Optional)* A font-palette-values rule matching this node.
        11. **parentLayoutNodeId** - *(Optional)* Id of the first parent element that does not have display: contents.
    """
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'CSS.getMatchedStylesForNode', 'params': params}
    json = (yield cmd_dict)
    return (CSSStyle.from_json(json['inlineStyle']) if 'inlineStyle' in json else None, CSSStyle.from_json(json['attributesStyle']) if 'attributesStyle' in json else None, [RuleMatch.from_json(i) for i in json['matchedCSSRules']] if 'matchedCSSRules' in json else None, [PseudoElementMatches.from_json(i) for i in json['pseudoElements']] if 'pseudoElements' in json else None, [InheritedStyleEntry.from_json(i) for i in json['inherited']] if 'inherited' in json else None, [InheritedPseudoElementMatches.from_json(i) for i in json['inheritedPseudoElements']] if 'inheritedPseudoElements' in json else None, [CSSKeyframesRule.from_json(i) for i in json['cssKeyframesRules']] if 'cssKeyframesRules' in json else None, [CSSPositionFallbackRule.from_json(i) for i in json['cssPositionFallbackRules']] if 'cssPositionFallbackRules' in json else None, [CSSPropertyRule.from_json(i) for i in json['cssPropertyRules']] if 'cssPropertyRules' in json else None, [CSSPropertyRegistration.from_json(i) for i in json['cssPropertyRegistrations']] if 'cssPropertyRegistrations' in json else None, CSSFontPaletteValuesRule.from_json(json['cssFontPaletteValuesRule']) if 'cssFontPaletteValuesRule' in json else None, dom.NodeId.from_json(json['parentLayoutNodeId']) if 'parentLayoutNodeId' in json else None)