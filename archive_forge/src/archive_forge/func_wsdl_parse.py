from __future__ import unicode_literals
import sys
import copy
import hashlib
import logging
import os
import tempfile
import warnings
from . import __author__, __copyright__, __license__, __version__, TIMEOUT
from .simplexml import SimpleXMLElement, TYPE_MAP, REVERSE_TYPE_MAP, Struct
from .transport import get_http_wrapper, set_http_wrapper, get_Http
from .helpers import Alias, fetch, sort_dict, make_key, process_element, \
from .wsse import UsernameToken
def wsdl_parse(self, url, cache=False):
    """Parse Web Service Description v1.1"""
    log.debug('Parsing wsdl url: %s' % url)
    force_download = False
    if cache:
        filename_pkl = '%s.pkl' % hashlib.md5(url).hexdigest()
        if isinstance(cache, basestring):
            filename_pkl = os.path.join(cache, filename_pkl)
        if os.path.exists(filename_pkl):
            log.debug('Unpickle file %s' % (filename_pkl,))
            f = open(filename_pkl, 'r')
            pkl = pickle.load(f)
            f.close()
            if pkl['version'][:-1] != __version__.split(' ')[0][:-1] or pkl['url'] != url:
                warnings.warn('version or url mismatch! discarding cached wsdl', RuntimeWarning)
                log.debug('Version: %s %s' % (pkl['version'], __version__))
                log.debug('URL: %s %s' % (pkl['url'], url))
                force_download = True
            else:
                self.namespace = pkl['namespace']
                self.documentation = pkl['documentation']
                return pkl['services']
    REVERSE_TYPE_MAP['string'] = str
    wsdl = self._url_to_xml_tree(url, cache, force_download)
    services = self._xml_tree_to_services(wsdl, cache, force_download)
    if cache:
        f = open(filename_pkl, 'wb')
        pkl = {'version': __version__.split(' ')[0], 'url': url, 'namespace': self.namespace, 'documentation': self.documentation, 'services': services}
        pickle.dump(pkl, f)
        f.close()
    return services