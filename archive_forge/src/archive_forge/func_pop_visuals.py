from __future__ import annotations
import logging # isort:skip
import sys
from collections.abc import Iterable
import numpy as np
from ..core.properties import ColorSpec
from ..models import ColumnarDataSource, ColumnDataSource, GlyphRenderer
from ..util.strings import nice_join
from ._legends import pop_legend_kwarg, update_legend
def pop_visuals(glyphclass, props, prefix='', defaults={}, override_defaults={}):
    """
    Applies basic cascading logic to deduce properties for a glyph.

    Args:
        glyphclass :
            the type of glyph being handled

        props (dict) :
            Maps properties and prefixed properties to their values.
            Keys in `props` matching `glyphclass` visual properties (those of
            'line_', 'fill_', 'hatch_' or 'text_') with added `prefix` will get
            popped, other keys will be ignored.
            Keys take the form '[{prefix}][{feature}_]{trait}'. Only {feature}
              must not contain underscores.
            Keys of the form '{prefix}{trait}' work as lower precedence aliases
              for {trait} for all {features}, as long as the glyph has no
              property called {trait}. I.e. this won't apply to "width" in a
              `rect` glyph.
            Ex: {'fill_color': 'blue', 'selection_line_width': 0.5}

        prefix (str) :
            Prefix used when accessing `props`. Ex: 'selection_'

        override_defaults (dict) :
            Explicitly provided fallback based on '{trait}', in case property
            not set in `props`.
            Ex. 'width' here may be used for 'selection_line_width'.

        defaults (dict) :
            Property fallback, in case prefixed property not in `props` or
            `override_defaults`.
            Ex. 'line_width' here may be used for 'selection_line_width'.

    Returns:
        result (dict) :
            Resulting properties for the instance (no prefixes).

    Notes:
        Feature trait 'text_color', as well as traits 'color' and 'alpha', have
        ultimate defaults in case those can't be deduced.
    """
    defaults = defaults.copy()
    defaults.setdefault('text_color', 'black')
    defaults.setdefault('hatch_color', 'black')
    trait_defaults = {}
    trait_defaults.setdefault('color', get_default_color())
    trait_defaults.setdefault('alpha', 1.0)
    result, traits = (dict(), set())
    prop_names = set(glyphclass.properties())
    for name in filter(_is_visual, prop_names):
        _, trait = _split_feature_trait(name)
        if prefix + name in props:
            result[name] = props.pop(prefix + name)
        elif trait not in prop_names and prefix + trait in props:
            result[name] = props[prefix + trait]
        elif trait in override_defaults:
            result[name] = override_defaults[trait]
        elif name in defaults:
            result[name] = defaults[name]
        elif trait in trait_defaults:
            result[name] = trait_defaults[trait]
        if trait not in prop_names:
            traits.add(trait)
    for trait in traits:
        props.pop(prefix + trait, None)
    return result