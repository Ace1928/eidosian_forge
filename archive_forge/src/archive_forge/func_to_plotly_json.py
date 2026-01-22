import abc
import collections
import inspect
import sys
import uuid
import random
from .._utils import patch_collections_abc, stringify_id, OrderedSet
def to_plotly_json(self):
    props = {p: getattr(self, p) for p in self._prop_names if hasattr(self, p)}
    props.update({k: getattr(self, k) for k in self.__dict__ if any((k.startswith(w) for w in self._valid_wildcard_attributes))})
    as_json = {'props': props, 'type': self._type, 'namespace': self._namespace}
    return as_json