import logging
import urllib.parse
import smart_open.utils
from ftplib import FTP, FTP_TLS, error_reply
import types
def convert_transport_params_to_args(transport_params):
    supported_keywords = ['timeout', 'source_address', 'encoding']
    unsupported_keywords = [k for k in transport_params if k not in supported_keywords]
    kwargs = {k: v for k, v in transport_params.items() if k in supported_keywords}
    if unsupported_keywords:
        logger.warning('ignoring unsupported ftp keyword arguments: %r', unsupported_keywords)
    return kwargs