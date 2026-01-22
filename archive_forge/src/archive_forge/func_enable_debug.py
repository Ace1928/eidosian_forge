import os
import re
import time
from typing import (
from urllib import parse
import requests
import gitlab
import gitlab.config
import gitlab.const
import gitlab.exceptions
from gitlab import _backends, utils
def enable_debug(self, mask_credentials: bool=True) -> None:
    import logging
    from http import client
    client.HTTPConnection.debuglevel = 1
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    httpclient_log = logging.getLogger('http.client')
    httpclient_log.propagate = True
    httpclient_log.setLevel(logging.DEBUG)
    requests_log = logging.getLogger('requests.packages.urllib3')
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True

    def print_as_log(*args: Any) -> None:
        httpclient_log.log(logging.DEBUG, ' '.join(args))
    setattr(client, 'print', print_as_log)
    if not mask_credentials:
        return
    token = self.private_token or self.oauth_token or self.job_token
    handler = logging.StreamHandler()
    handler.setFormatter(utils.MaskingFormatter(masked=token))
    logger.handlers.clear()
    logger.addHandler(handler)