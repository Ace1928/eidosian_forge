from __future__ import annotations
import http
import logging
import sys
from copy import copy
from typing import Literal
import click
def color_level_name(self, level_name: str, level_no: int) -> str:

    def default(level_name: str) -> str:
        return str(level_name)
    func = self.level_name_colors.get(level_no, default)
    return func(level_name)