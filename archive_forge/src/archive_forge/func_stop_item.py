from collections import defaultdict, deque
from typing import Any, Counter, Dict, Iterable, Iterator, List, \
from ..exceptions import XMLSchemaValueError
from ..aliases import ModelGroupType, ModelParticleType, SchemaElementType
from ..translation import gettext as _
from .. import limits
from .exceptions import XMLSchemaModelError, XMLSchemaModelDepthError
from .wildcards import XsdAnyElement, Xsd11AnyElement
from . import groups
def stop_item(item: ModelParticleType) -> bool:
    """
            Stops element or group matching, incrementing current group counter.

            :return: `True` if the item has violated the minimum occurrences for itself             or for the current group, `False` otherwise.
            """
    if isinstance(item, groups.XsdGroup):
        self.group, self.items, self.match = self._groups.pop()
    if self.group.model == 'choice':
        item_occurs = occurs[item]
        if not item_occurs:
            return False
        item_max_occurs = occurs[item,] or item_occurs
        if item.max_occurs is None:
            min_group_occurs = 1
        elif item_occurs % item.max_occurs:
            min_group_occurs = 1 + item_occurs // item.max_occurs
        else:
            min_group_occurs = item_occurs // item.max_occurs
        max_group_occurs = max(1, item_max_occurs // (item.min_occurs or 1))
        occurs[self.group] += min_group_occurs
        occurs[self.group,] += max_group_occurs
        occurs[item] = 0
        self.items = self.iter_group()
        self.match = False
        return item.is_missing(item_max_occurs)
    elif self.group.model == 'all':
        return False
    elif self.match:
        pass
    elif occurs[item]:
        self.match = True
    elif item.is_emptiable():
        return False
    elif self._groups:
        return stop_item(self.group)
    elif self.group.min_occurs <= max(occurs[self.group], occurs[self.group,]):
        return stop_item(self.group)
    else:
        return True
    if item is self.group[-1]:
        for k, item2 in enumerate(self.group, start=1):
            item_occurs = occurs[item2]
            if not item_occurs:
                continue
            item_max_occurs = occurs[item2,] or item_occurs
            if item_max_occurs == 1 or any((not x.is_emptiable() for x in self.group[k:])):
                occurs[self.group] += 1
                break
            min_group_occurs = max(1, item_occurs // (item2.max_occurs or item_occurs))
            max_group_occurs = max(1, item_max_occurs // (item2.min_occurs or 1))
            occurs[self.group] += min_group_occurs
            occurs[self.group,] += max_group_occurs
            break
    return item.is_missing(max(occurs[item], occurs[item,]))