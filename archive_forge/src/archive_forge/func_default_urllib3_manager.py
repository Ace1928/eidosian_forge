import logging
import os
import select
import socket
import subprocess
import sys
from contextlib import closing
from io import BufferedReader, BytesIO
from typing import (
from urllib.parse import quote as urlquote
from urllib.parse import unquote as urlunquote
from urllib.parse import urljoin, urlparse, urlunparse, urlunsplit
import dulwich
from .config import Config, apply_instead_of, get_xdg_config_home_path
from .errors import GitProtocolError, NotGitRepository, SendPackError
from .pack import (
from .protocol import (
from .refs import PEELED_TAG_SUFFIX, _import_remote_refs, read_info_refs
from .repo import Repo
def default_urllib3_manager(config, pool_manager_cls=None, proxy_manager_cls=None, base_url=None, **override_kwargs) -> Union['urllib3.ProxyManager', 'urllib3.PoolManager']:
    """Return urllib3 connection pool manager.

    Honour detected proxy configurations.

    Args:
      config: `dulwich.config.ConfigDict` instance with Git configuration.
      override_kwargs: Additional arguments for `urllib3.ProxyManager`

    Returns:
      Either pool_manager_cls (defaults to `urllib3.ProxyManager`) instance for
      proxy configurations, proxy_manager_cls
      (defaults to `urllib3.PoolManager`) instance otherwise

    """
    proxy_server = user_agent = None
    ca_certs = ssl_verify = None
    if proxy_server is None:
        for proxyname in ('https_proxy', 'http_proxy', 'all_proxy'):
            proxy_server = os.environ.get(proxyname)
            if proxy_server:
                break
    if proxy_server:
        if check_for_proxy_bypass(base_url):
            proxy_server = None
    if config is not None:
        if proxy_server is None:
            try:
                proxy_server = config.get(b'http', b'proxy')
            except KeyError:
                pass
        try:
            user_agent = config.get(b'http', b'useragent')
        except KeyError:
            pass
        try:
            ssl_verify = config.get_boolean(b'http', b'sslVerify')
        except KeyError:
            ssl_verify = True
        try:
            ca_certs = config.get(b'http', b'sslCAInfo')
        except KeyError:
            ca_certs = None
    if user_agent is None:
        user_agent = default_user_agent_string()
    headers = {'User-agent': user_agent}
    kwargs = {'ca_certs': ca_certs}
    if ssl_verify is True:
        kwargs['cert_reqs'] = 'CERT_REQUIRED'
    elif ssl_verify is False:
        kwargs['cert_reqs'] = 'CERT_NONE'
    else:
        kwargs['cert_reqs'] = 'CERT_REQUIRED'
    kwargs.update(override_kwargs)
    import urllib3
    if proxy_server is not None:
        if proxy_manager_cls is None:
            proxy_manager_cls = urllib3.ProxyManager
        if not isinstance(proxy_server, str):
            proxy_server = proxy_server.decode()
        proxy_server_url = urlparse(proxy_server)
        if proxy_server_url.username is not None:
            proxy_headers = urllib3.make_headers(proxy_basic_auth=f'{proxy_server_url.username}:{proxy_server_url.password or ''}')
        else:
            proxy_headers = {}
        manager = proxy_manager_cls(proxy_server, proxy_headers=proxy_headers, headers=headers, **kwargs)
    else:
        if pool_manager_cls is None:
            pool_manager_cls = urllib3.PoolManager
        manager = pool_manager_cls(headers=headers, **kwargs)
    return manager