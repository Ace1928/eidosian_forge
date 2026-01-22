from dogpile.cache import api
from dogpile.cache import proxy
from oslo_context import context as oslo_context
from oslo_serialization import msgpackutils
def set_multi(self, mapping):
    for k, v in mapping.items():
        self._set_local_cache(k, v)
    self.proxied.set_multi(mapping)