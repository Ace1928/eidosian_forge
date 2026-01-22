from __future__ import annotations
import typing
from copy import deepcopy
from dataclasses import dataclass
from ._plot_side_space import LRTBSpaces, WHSpaceParts, calculate_panel_spacing
from .utils import bbox_in_figure_space, get_transPanels
def vertically_align_text_with_panels(text: Text, params: GridSpecParams, va: str | float, pack: LayoutPack):
    """
    Vertical justification

    Reinterpret vertical alignment to be justification about the panels.
    """
    if isinstance(va, str):
        lookup = {'top': 1.0, 'center': 0.5, 'baseline': 0.5, 'center_baseline': 0.5, 'bottom': 0.0}
        f = lookup[va]
    else:
        f = va
    box = bbox_in_figure_space(text, pack.figure, pack.renderer)
    y = params.bottom * (1 - f) + (params.top - box.height) * f
    text.set_y(y)
    text.set_verticalalignment('bottom')