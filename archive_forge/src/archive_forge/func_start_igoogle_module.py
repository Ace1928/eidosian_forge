from __future__ import annotations
import copy
import typing as t
from . import common
def start_igoogle_module(self, attrs: dict[str, str]) -> None:
    if attrs.get('type', '').strip().lower() == 'rss':
        self.flag_feed = True