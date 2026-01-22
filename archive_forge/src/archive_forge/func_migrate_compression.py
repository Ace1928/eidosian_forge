from typing import Any, Dict, Optional, Union, Iterable, List, Type, TYPE_CHECKING
from lazyops.utils.lazy import get_keydb_session
from .base import BaseStatefulBackend, SchemaType, logger
def migrate_compression(self, **kwargs) -> List[str]:
    """
        Migrates the Compression
        """
    if not self.hset_enabled and (not self.base_key):
        raise NotImplementedError('Cannot get all data from a Redis Cache without a base key')
    failed_keys = []
    null_is_error = self.serializer.raise_errors
    if self.hset_enabled:
        data = self.cache.hgetall(self.base_key)
        results = {}
        for key, value in data.items():
            if isinstance(key, bytes):
                key = key.decode()
            try:
                val = self.decode_value(value)
                if val is not None:
                    results[key] = val
                elif null_is_error:
                    failed_keys.append(key)
            except Exception as e:
                if not null_is_error:
                    logger.warning(f'Unable to decode value for {key}: {e}')
                failed_keys.append(key)
        if results:
            self.set_batch(results)
        if failed_keys:
            logger.warning(f'Failed to migrate keys: {failed_keys}')
        return failed_keys
    keys = self._fetch_set_keys(decode=True)
    data_list = self.cache.client.mget(keys)
    results: Dict[str, Any] = {}
    for key, value in zip(keys, data_list):
        if isinstance(key, bytes):
            key = key.decode()
        try:
            val = self.decode_value(value)
            if val is not None:
                results[key] = val
            elif null_is_error:
                failed_keys.append(key)
        except Exception as e:
            if not null_is_error:
                logger.warning(f'Unable to decode value for {key}: {e}')
            failed_keys.append(key)
    if results:
        self.set_batch(results)
    if failed_keys:
        logger.warning(f'Failed to migrate keys: {failed_keys}')
    return failed_keys