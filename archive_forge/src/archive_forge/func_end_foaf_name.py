from __future__ import annotations
import copy
import typing as t
from . import common
def end_foaf_name(self) -> None:
    value = self.get_text()
    if self.flag_feed and self.flag_new_title:
        self.foaf_name.append(value)
        self.flag_new_title = False
    elif self.flag_group and value:
        self.hierarchy.append(value)
        self.flag_group = False