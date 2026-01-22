from __future__ import annotations
import asyncio
import random
import inspect
import aiohttpx
import functools
import subprocess
from pydantic import BaseModel
from urllib.parse import urlparse
from lazyops.libs.proxyobj import ProxyObject, proxied
from .base import BaseGlobalClient, cachify
from .utils import aget_root_domain, get_user_agent, http_retry_wrapper
from typing import Optional, Type, TypeVar, Literal, Union, Set, Awaitable, Any, Dict, List, Callable, overload, TYPE_CHECKING
def get_csx(self, query: str, exact_terms: Optional[str]=None, exclude_terms: Optional[str]=None, file_type: Optional[str]=None, cachable: Optional[bool]=True, overwrite_cache: Optional[bool]=None, disable_cache: Optional[bool]=None, background: Optional[bool]=False, callback: Optional[Callable[..., Any]]=None, **kwargs) -> Dict[str, Union[List[Dict[str, Any]], Any]]:
    """
        Returns the Google CSX Results

        file_type: pdf
        """
    if not self.google_csx_api_key and (not self.google_csx_id):
        raise ValueError('Google CSX API Key and ID are not set')
    params = {'key': self.google_csx_api_key, 'cx': self.google_csx_id, 'q': query}
    if exact_terms:
        params['exactTerms'] = exact_terms
    if exclude_terms:
        params['excludeTerms'] = exclude_terms
    if file_type:
        if 'application/' not in file_type:
            file_type = f'application/{file_type}'
        params['fileType'] = file_type
    if kwargs:
        params.update(kwargs)
    if background:
        return self.pooler.create_background(self.get, url=self.google_csx_base_url, return_type='json', params=params, task_callback=callback, cachable=cachable, overwrite_cache=overwrite_cache, disable_cache=disable_cache)
    return self.get(url=self.google_csx_base_url, return_type='json', params=params, cachable=cachable, overwrite_cache=overwrite_cache, disable_cache=disable_cache)