from __future__ import annotations
import copy
import typing as t
from . import common
def start_foaf_agent(self, _: t.Any) -> None:
    self.flag_agent = True
    self.flag_feed = True
    self.flag_new_title = True