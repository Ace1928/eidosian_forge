import asyncio
import json
import logging
import numbers
import socket
import sys
from urllib.parse import quote, unquote, urljoin, urlparse
from tornado import httpclient, ioloop
@classmethod
def validate_http_api(cls, http_api):
    url = urlparse(http_api)
    if url.scheme not in ('http', 'https'):
        raise ValueError(f'Invalid http api schema: {url.scheme}')