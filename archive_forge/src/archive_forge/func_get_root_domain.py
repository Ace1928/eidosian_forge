import random
import inspect
import backoff
import functools
import aiohttpx
from typing import Callable, List, Optional, TypeVar, TYPE_CHECKING
@functools.lru_cache()
def get_root_domain(url: str, attempts: Optional[int]=None) -> str:
    """
    Returns the root domain after redirection
    """
    if not url.startswith('http'):
        url = f'https://{url}'
    try:
        with aiohttpx.Client(follow_redirects=True, verify=False) as client:
            r = client.get(url, timeout=5)
        r.raise_for_status()
        return str(r.url).rstrip('/')
    except (aiohttpx.ConnectTimeout, aiohttpx.ReadTimeout) as e:
        return url
    except aiohttpx.HTTPStatusError as e:
        return str(e.response.url).rstrip('/') if e.response.status_code < 500 else url
    except Exception as e:
        attempts = attempts + 1 if attempts else 1
        if attempts > 2:
            return None
        new_url = f'https://{extract_domain(url)}'
        return get_root_domain(new_url, attempts=attempts)