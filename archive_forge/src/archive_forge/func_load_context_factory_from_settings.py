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
def load_context_factory_from_settings(settings, crawler):
    ssl_method = openssl_methods[settings.get('DOWNLOADER_CLIENT_TLS_METHOD')]
    context_factory_cls = load_object(settings['DOWNLOADER_CLIENTCONTEXTFACTORY'])
    try:
        context_factory = create_instance(objcls=context_factory_cls, settings=settings, crawler=crawler, method=ssl_method)
    except TypeError:
        context_factory = create_instance(objcls=context_factory_cls, settings=settings, crawler=crawler)
        msg = f'{settings['DOWNLOADER_CLIENTCONTEXTFACTORY']} does not accept a `method` argument (type OpenSSL.SSL method, e.g. OpenSSL.SSL.SSLv23_METHOD) and/or a `tls_verbose_logging` argument and/or a `tls_ciphers` argument. Please, upgrade your context factory class to handle them or ignore them.'
        warnings.warn(msg)
    return context_factory