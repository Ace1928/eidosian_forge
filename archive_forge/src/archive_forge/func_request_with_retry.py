import base64
import functools
import itertools
import logging
import os
import queue
import random
import sys
import threading
import time
from types import TracebackType
from typing import (
import requests
import wandb
from wandb import util
from wandb.sdk.internal import internal_api
from ..lib import file_stream_utils
def request_with_retry(func: Callable, *args: Any, **kwargs: Any) -> Union['requests.Response', 'requests.RequestException']:
    """Perform a requests http call, retrying with exponential backoff.

    Arguments:
        func:        An http-requesting function to call, like requests.post
        max_retries: Maximum retries before giving up.
                     By default, we retry 30 times in ~2 hours before dropping the chunk
        *args:       passed through to func
        **kwargs:    passed through to func
    """
    max_retries: int = kwargs.pop('max_retries', 30)
    retry_callback: Optional[Callable] = kwargs.pop('retry_callback', None)
    sleep = 2
    retry_count = 0
    while True:
        try:
            response: requests.Response = func(*args, **kwargs)
            response.raise_for_status()
            return response
        except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError, requests.exceptions.Timeout) as e:
            if isinstance(e, requests.exceptions.HTTPError):
                if e.response is not None and e.response.status_code in {400, 403, 404, 409}:
                    return e
            if retry_count == max_retries:
                return e
            retry_count += 1
            delay = sleep + random.random() * 0.25 * sleep
            if isinstance(e, requests.exceptions.HTTPError) and (e.response is not None and e.response.status_code == 429):
                err_str = f'Filestream rate limit exceeded, retrying in {delay:.1f} seconds. '
                if retry_callback:
                    retry_callback(e.response.status_code, err_str)
                logger.info(err_str)
            else:
                logger.warning('requests_with_retry encountered retryable exception: %s. func: %s, args: %s, kwargs: %s', e, func, args, kwargs)
            time.sleep(delay)
            sleep *= 2
            if sleep > MAX_SLEEP_SECONDS:
                sleep = MAX_SLEEP_SECONDS
        except requests.exceptions.RequestException as e:
            error_message = 'unknown error'
            try:
                error_message = response.json()['error']
            except Exception:
                pass
            logger.error(f'requests_with_retry error: {error_message}')
            logger.exception('requests_with_retry encountered unretryable exception: %s', e)
            return e