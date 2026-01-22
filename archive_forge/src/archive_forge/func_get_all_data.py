from typing import Any, Dict, Optional, Union, Iterable, List, Type, TYPE_CHECKING
from lazyops.utils.lazy import get_keydb_session
from .base import BaseStatefulBackend, SchemaType, logger
def get_all_data(self, exclude_base_key: Optional[bool]=False, **kwargs) -> Dict[str, Any]:
    """
        Loads all the Data
        """
    if not self.hset_enabled and (not self.base_key):
        raise NotImplementedError('Cannot get all data from a Redis Cache without a base key')
    if self.hset_enabled:
        data = self.cache.hgetall(self.base_key)
        results = {}
        for key, value in data.items():
            if isinstance(key, bytes):
                key = key.decode()
            try:
                results[key] = self.decode_value(value, **kwargs)
            except AttributeError:
                logger.warning(f'Unable to decode value for {key}')
                self.delete(key)
        return results
    keys = self._fetch_set_keys(decode=True)
    data_list = self.cache.client.mget(keys)
    results: Dict[str, Any] = {}
    for key, value in zip(keys, data_list):
        if isinstance(key, bytes):
            key = key.decode()
        try:
            results[key] = self.decode_value(value, **kwargs)
        except AttributeError:
            logger.warning(f'Unable to decode value for {key}')
            self.delete(key)
    if exclude_base_key:
        results = {k.replace(f'{self.base_key}.', ''): v for k, v in results.items()}
    return results