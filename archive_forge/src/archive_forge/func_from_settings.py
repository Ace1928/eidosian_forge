import warnings
from typing import TYPE_CHECKING, Any, List, Optional
from OpenSSL import SSL
from twisted.internet._sslverify import _setAcceptableProtocols
from twisted.internet.ssl import (
from twisted.web.client import BrowserLikePolicyForHTTPS
from twisted.web.iweb import IPolicyForHTTPS
from zope.interface.declarations import implementer
from zope.interface.verify import verifyObject
from scrapy.core.downloader.tls import (
from scrapy.settings import BaseSettings
from scrapy.utils.misc import create_instance, load_object
@classmethod
def from_settings(cls, settings: BaseSettings, method: int=SSL.SSLv23_METHOD, *args: Any, **kwargs: Any):
    tls_verbose_logging: bool = settings.getbool('DOWNLOADER_CLIENT_TLS_VERBOSE_LOGGING')
    tls_ciphers: Optional[str] = settings['DOWNLOADER_CLIENT_TLS_CIPHERS']
    return cls(*args, method=method, tls_verbose_logging=tls_verbose_logging, tls_ciphers=tls_ciphers, **kwargs)