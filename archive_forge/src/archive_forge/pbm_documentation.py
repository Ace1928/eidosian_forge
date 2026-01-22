import logging
import os
import urllib.parse as urlparse
import urllib.request as urllib
from oslo_vmware import service
from oslo_vmware import vim_util
Set the specified vCenter session cookie in the SOAP header

        :param cookie: cookie to set
        