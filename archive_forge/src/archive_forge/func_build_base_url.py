import http.client as httplib
import io
import logging
import netaddr
from oslo_utils import timeutils
from oslo_utils import uuidutils
import requests
import suds
from suds import cache
from suds import client
from suds import plugin
import suds.sax.element as element
from suds import transport
from oslo_vmware._i18n import _
from oslo_vmware import exceptions
from oslo_vmware import vim_util
@staticmethod
def build_base_url(protocol, host, port):
    proto_str = '%s://' % protocol
    host_str = '[%s]' % host if netaddr.valid_ipv6(host) else host
    port_str = '' if port is None else ':%d' % port
    return proto_str + host_str + port_str