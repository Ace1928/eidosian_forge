from lazyops import get_logger, timer
from lazyops.apis import LazyAPI
def get_base_params(self, is_fallback=False):
    if is_fallback:
        if self._fallback == 'eai':
            return self._eai_config.copy()
        return self._ts_config.copy()
    if self._fallback == 'eai':
        return self._ts_config.copy()
    return self._eai_config.copy()