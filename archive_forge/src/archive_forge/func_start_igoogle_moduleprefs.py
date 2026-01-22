from __future__ import annotations
import copy
import typing as t
from . import common
def start_igoogle_moduleprefs(self, attrs: dict[str, str]) -> None:
    if self.flag_feed and attrs.get('xmlurl', '').strip():
        obj = common.SuperDict({'url': attrs['xmlurl'].strip()})
        obj['title'] = ''
        if self.hierarchy:
            obj['categories'] = [copy.copy(self.hierarchy)]
        if len(self.hierarchy) == 1:
            obj['tags'] = copy.copy(self.hierarchy)
        self.harvest['feeds'].append(obj)