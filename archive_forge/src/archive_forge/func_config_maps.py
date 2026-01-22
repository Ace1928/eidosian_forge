from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from typing import Mapping, Sequence
from googlecloudsdk.api_lib.run import k8s_object
@property
def config_maps(self):
    """Mutable dict-like object for mounts whose volumes have a config map source type."""
    return k8s_object.KeyValueListAsDictionaryWrapper(self._m, self._item_class, key_field=self._key_field, value_field=self._value_field, filter_func=lambda mount: mount.name in self._volumes.config_maps)