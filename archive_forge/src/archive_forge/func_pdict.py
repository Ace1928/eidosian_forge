from __future__ import annotations
from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Any, Dict, Optional, Iterable, Type, TYPE_CHECKING
@property
def pdict(self) -> 'PersistentDict':
    """
        Returns the Persistent Dict
        """
    if self._pdict is None:
        from lazyops.libs.authzero.utils.lazy import get_az_pdict
        from lazyops.libs.authzero.utils.helpers import normalize_audience_name
        base_key = f'{self.settings.base_cache_key}.'
        if self.settings.app_name:
            base_key += f'{self.settings.app_name}.'
        elif self.settings.app_ingress:
            base_key += f'{normalize_audience_name(self.settings.app_ingress)}.'
        else:
            base_key += 'default.'
        base_key += f'{self.settings.app_env.name}.{self.name}'
        base_key = base_key.replace(' ', '_').lower().replace('..', '.')
        self._pdict = get_az_pdict(base_key=base_key, **self._pdict_kwargs)
    return self._pdict