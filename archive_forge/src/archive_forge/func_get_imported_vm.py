import logging
import ssl
import time
from oslo_utils import excutils
from oslo_utils import netutils
import requests
import urllib.parse as urlparse
from urllib3 import connection as httplib
from oslo_vmware._i18n import _
from oslo_vmware import exceptions
from oslo_vmware import vim_util
def get_imported_vm(self):
    """"Get managed object reference of the VM created for import.

        :raises: VimException
        """
    if self._get_progress() < 100:
        excep_msg = _('Incomplete VMDK upload to %s.') % self._url
        LOG.exception(excep_msg)
        raise exceptions.ImageTransferException(excep_msg)
    return self._vm_ref