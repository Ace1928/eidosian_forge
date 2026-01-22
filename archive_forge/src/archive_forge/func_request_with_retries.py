from __future__ import annotations
import collections
import datetime
import functools
import importlib
import importlib.metadata
import io
import json
import logging
import os
import random
import re
import socket
import sys
import threading
import time
import uuid
import warnings
import weakref
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue, Queue
from typing import (
from urllib import parse as urllib_parse
import orjson
import requests
from requests import adapters as requests_adapters
from urllib3.util import Retry
import langsmith
from langsmith import env as ls_env
from langsmith import schemas as ls_schemas
from langsmith import utils as ls_utils
def request_with_retries(self, /, method: Literal['GET', 'POST', 'PUT', 'PATCH', 'DELETE'], pathname: str, *, request_kwargs: Optional[Mapping]=None, stop_after_attempt: int=1, retry_on: Optional[Sequence[Type[BaseException]]]=None, to_ignore: Optional[Sequence[Type[BaseException]]]=None, handle_response: Optional[Callable[[requests.Response, int], Any]]=None, **kwargs: Any) -> requests.Response:
    """Send a request with retries.

        Parameters
        ----------
        request_method : str
            The HTTP request method.
        pathname : str
            The pathname of the request URL. Will be appended to the API URL.
        request_kwargs : Mapping
            Additional request parameters.
        stop_after_attempt : int, default=1
            The number of attempts to make.
        retry_on : Sequence[Type[BaseException]] or None, default=None
            The exceptions to retry on. In addition to:
            [LangSmithConnectionError, LangSmithAPIError].
        to_ignore : Sequence[Type[BaseException]] or None, default=None
            The exceptions to ignore / pass on.
        handle_response : Callable[[requests.Response, int], Any] or None, default=None
            A function to handle the response and return whether to continue
            retrying.
        **kwargs : Any
            Additional keyword arguments to pass to the request.

        Returns:
        -------
        Response
            The response object.

        Raises:
        ------
        LangSmithAPIError
            If a server error occurs.
        LangSmithUserError
            If the request fails.
        LangSmithConnectionError
            If a connection error occurs.
        LangSmithError
            If the request fails.
        """
    request_kwargs = request_kwargs or {}
    request_kwargs = {'headers': {**self._headers, **request_kwargs.get('headers', {}), **kwargs.get('headers', {})}, 'timeout': (self.timeout_ms[0] / 1000, self.timeout_ms[1] / 1000), **request_kwargs, **kwargs}
    if method != 'GET' and 'data' in request_kwargs and (not request_kwargs['headers'].get('Content-Type')):
        request_kwargs['headers']['Content-Type'] = 'application/json'
    logging_filters = [ls_utils.FilterLangSmithRetry(), ls_utils.FilterPoolFullWarning(host=str(self._host))]
    retry_on_: Tuple[Type[BaseException], ...] = (*(retry_on or []), *(ls_utils.LangSmithConnectionError, ls_utils.LangSmithAPIError))
    to_ignore_: Tuple[Type[BaseException], ...] = (*(to_ignore or ()),)
    response = None
    for idx in range(stop_after_attempt):
        try:
            try:
                with ls_utils.filter_logs(_urllib3_logger, logging_filters):
                    response = self.session.request(method, self.api_url + pathname if not pathname.startswith('http') else pathname, stream=False, **request_kwargs)
                ls_utils.raise_for_status_with_text(response)
                return response
            except requests.exceptions.ReadTimeout as e:
                logger.debug('Passing on exception %s', e)
                if idx + 1 == stop_after_attempt:
                    raise
                sleep_time = 2 ** idx + random.random() * 0.5
                time.sleep(sleep_time)
                continue
            except requests.HTTPError as e:
                if response is not None:
                    if handle_response is not None:
                        if idx + 1 < stop_after_attempt:
                            should_continue = handle_response(response, idx + 1)
                            if should_continue:
                                continue
                    if response.status_code == 500:
                        raise ls_utils.LangSmithAPIError(f'Server error caused failure to {method} {pathname} in LangSmith API. {repr(e)}')
                    elif response.status_code == 429:
                        raise ls_utils.LangSmithRateLimitError(f'Rate limit exceeded for {pathname}. {repr(e)}')
                    elif response.status_code == 401:
                        raise ls_utils.LangSmithAuthError(f'Authentication failed for {pathname}. {repr(e)}')
                    elif response.status_code == 404:
                        raise ls_utils.LangSmithNotFoundError(f'Resource not found for {pathname}. {repr(e)}')
                    elif response.status_code == 409:
                        raise ls_utils.LangSmithConflictError(f'Conflict for {pathname}. {repr(e)}')
                    else:
                        raise ls_utils.LangSmithError(f'Failed to {method} {pathname} in LangSmith API. {repr(e)}')
                else:
                    raise ls_utils.LangSmithUserError(f'Failed to {method} {pathname} in LangSmith API. {repr(e)}')
            except requests.ConnectionError as e:
                recommendation = 'Please confirm your LANGCHAIN_ENDPOINT' if self.api_url != 'https://api.smith.langchain.com' else 'Please confirm your internet connection.'
                raise ls_utils.LangSmithConnectionError(f'Connection error caused failure to {method} {pathname}  in LangSmith API. {recommendation}. {repr(e)}') from e
            except Exception as e:
                args = list(e.args)
                msg = args[1] if len(args) > 1 else ''
                msg = msg.replace('session', 'session (project)')
                emsg = '\n'.join([str(args[0])] + [msg] + [str(arg) for arg in args[2:]])
                raise ls_utils.LangSmithError(f'Failed to {method} {pathname} in LangSmith API. {emsg}') from e
        except to_ignore_ as e:
            if response is not None:
                logger.debug('Passing on exception %s', e)
                return response
        except retry_on_:
            if idx + 1 == stop_after_attempt:
                raise
            sleep_time = 2 ** idx + random.random() * 0.5
            time.sleep(sleep_time)
            continue
    raise ls_utils.LangSmithError(f'Failed to {method} {pathname} in LangSmith API.')