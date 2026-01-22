from dogpile.cache import api
from dogpile.cache import proxy
from oslo_context import context as oslo_context
from oslo_serialization import msgpackutils
def get_multi(self, keys):
    values = {}
    for key in keys:
        v = self._get_local_cache(key)
        if v is not api.NO_VALUE:
            values[key] = v
    query_keys = set(keys).difference(set(values.keys()))
    values.update(dict(zip(query_keys, self.proxied.get_multi(query_keys))))
    return [values[k] for k in keys]