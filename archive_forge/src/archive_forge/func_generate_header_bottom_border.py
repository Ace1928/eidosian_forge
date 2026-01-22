import copy
import io
from collections import (
from enum import (
from typing import (
from wcwidth import (  # type: ignore[import]
from . import (
def generate_header_bottom_border(self) -> str:
    """Generate a border which appears at the bottom of the header"""
    fill_char = '═'
    pre_line = '╠' + self.padding * '═'
    inter_cell = self.padding * '═'
    if self.column_borders:
        inter_cell += '╪'
    inter_cell += self.padding * '═'
    post_line = self.padding * '═' + '╣'
    return self.generate_row(self.empty_data, is_header=False, fill_char=self.apply_border_color(fill_char), pre_line=self.apply_border_color(pre_line), inter_cell=self.apply_border_color(inter_cell), post_line=self.apply_border_color(post_line))