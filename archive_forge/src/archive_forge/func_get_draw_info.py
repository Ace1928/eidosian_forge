from __future__ import annotations
from typing import TYPE_CHECKING, List
from ..iapi import strip_draw_info, strip_label_details
def get_draw_info(self) -> strip_draw_info:
    """
        Get information required to draw strips

        Returns
        -------
        out :
            A structure with all the coordinates (x, y) required
            to draw the strip text and the background box
            (box_x, box_y, box_width, box_height).
        """
    theme = self.theme
    position = self.position
    if position == 'top':
        y = 1
        ha, va = ('center', 'bottom')
        rotation = theme.getp(('strip_text_x', 'rotation'))
        box_width = 1
        box_height = 0
        strip_text_margin = theme.getp(('strip_text_x', 'margin')).get_as('b', 'lines')
        strip_align = theme.getp('strip_align_x')
        x = theme.getp(('strip_text_x', 'x'), 0)
        box_width = theme.getp(('strip_background_x', 'width'), 1)
    elif position == 'right':
        x = 1
        ha, va = ('left', 'center')
        rotation = theme.getp(('strip_text_y', 'rotation'))
        box_width = 0
        strip_text_margin = theme.getp(('strip_text_y', 'margin')).get_as('r', 'lines')
        strip_align = theme.getp('strip_align_y')
        y = theme.getp(('strip_text_y', 'y'), 0)
        box_height = theme.getp(('strip_background_y', 'height'), 1)
    else:
        raise ValueError(f'Unknown position for strip text: {position!r}')
    return strip_draw_info(x=x, y=y, ha=ha, va=va, box_width=box_width, box_height=box_height, strip_text_margin=strip_text_margin, strip_align=strip_align, position=position, label=self.label_info.text(), ax=self.ax, rotation=rotation, layout=self.layout_info)