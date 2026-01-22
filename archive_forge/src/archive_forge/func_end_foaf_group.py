from __future__ import annotations
import copy
import typing as t
from . import common
def end_foaf_group(self) -> None:
    self.flag_group = False
    for key, obj in self.group_objs:
        if obj['url'] in self.found_urls:
            obj = self.found_urls[obj['url']][1]
        else:
            self.found_urls[obj['url']] = (key, obj)
            self.harvest[key].append(obj)
        obj.setdefault('categories', [])
        obj.setdefault('tags', [])
        if self.hierarchy and self.hierarchy not in obj['categories']:
            obj['categories'].append(copy.copy(self.hierarchy))
        if len(self.hierarchy) == 1 and self.hierarchy[0] not in obj['tags']:
            obj['tags'].extend(copy.copy(self.hierarchy))
    self.group_objs = []
    if self.hierarchy:
        self.hierarchy.pop()