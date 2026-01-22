import base64
import inspect
import logging
import socket
import subprocess
import uuid
from contextlib import closing
from itertools import islice
from sys import version_info
def get_results_from_paginated_fn(paginated_fn, max_results_per_page, max_results=None):
    """Gets results by calling the ``paginated_fn`` until either no more results remain or
    the specified ``max_results`` threshold has been reached.

    Args:
        paginated_fn: This function is expected to take in the number of results to retrieve
            per page and a pagination token, and return a PagedList object.
        max_results_per_page: The maximum number of results to retrieve per page.
        max_results: The maximum number of results to retrieve overall. If unspecified,
            all results will be retrieved.

    Returns:
        Returns a list of entities, as determined by the paginated_fn parameter, with no more
        entities than specified by max_results.

    """
    all_results = []
    next_page_token = None
    returns_all = max_results is None
    while returns_all or len(all_results) < max_results:
        num_to_get = max_results_per_page if returns_all else max_results - len(all_results)
        if num_to_get < max_results_per_page:
            page_results = paginated_fn(num_to_get, next_page_token)
        else:
            page_results = paginated_fn(max_results_per_page, next_page_token)
        all_results.extend(page_results)
        if hasattr(page_results, 'token') and page_results.token:
            next_page_token = page_results.token
        else:
            break
    return all_results