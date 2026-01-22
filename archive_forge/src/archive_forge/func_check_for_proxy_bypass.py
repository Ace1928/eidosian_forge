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
def check_for_proxy_bypass(base_url):
    if base_url:
        no_proxy_str = os.environ.get('no_proxy')
        if no_proxy_str:
            parsed_url = urlparse(base_url)
            hostname = parsed_url.hostname
            if hostname:
                import ipaddress
                try:
                    hostname_ip = ipaddress.ip_address(hostname)
                except ValueError:
                    hostname_ip = None
                no_proxy_values = no_proxy_str.split(',')
                for no_proxy_value in no_proxy_values:
                    no_proxy_value = no_proxy_value.strip()
                    if no_proxy_value:
                        no_proxy_value = no_proxy_value.lower()
                        no_proxy_value = no_proxy_value.lstrip('.')
                        if hostname_ip:
                            try:
                                no_proxy_value_network = ipaddress.ip_network(no_proxy_value, strict=False)
                            except ValueError:
                                no_proxy_value_network = None
                            if no_proxy_value_network:
                                if hostname_ip in no_proxy_value_network:
                                    return True
                        if no_proxy_value == '*':
                            return True
                        if hostname == no_proxy_value:
                            return True
                        no_proxy_value = '.' + no_proxy_value
                        if hostname.endswith(no_proxy_value):
                            return True
    return False