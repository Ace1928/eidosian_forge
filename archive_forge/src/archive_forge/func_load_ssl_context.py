from __future__ import annotations
import logging
import os
import ssl
import typing
from pathlib import Path
import certifi
from ._compat import set_minimum_tls_version_1_2
from ._models import Headers
from ._types import CertTypes, HeaderTypes, TimeoutTypes, URLTypes, VerifyTypes
from ._urls import URL
from ._utils import get_ca_bundle_from_env
def load_ssl_context(self) -> ssl.SSLContext:
    logger.debug('load_ssl_context verify=%r cert=%r trust_env=%r http2=%r', self.verify, self.cert, self.trust_env, self.http2)
    if self.verify:
        return self.load_ssl_context_verify()
    return self.load_ssl_context_no_verify()