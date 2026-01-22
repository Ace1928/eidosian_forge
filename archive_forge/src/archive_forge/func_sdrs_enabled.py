import logging
import posixpath
import random
import re
import http.client as httplib
import urllib.parse as urlparse
from oslo_vmware._i18n import _
from oslo_vmware import constants
from oslo_vmware import exceptions
from oslo_vmware import vim_util
def sdrs_enabled(session, dsc_ref):
    """Check if Storage DRS is enabled for the given datastore cluster.

    :param session: VMwareAPISession object
    :param dsc_ref: datastore cluster moref
    """
    pod_sdrs_entry = session.invoke_api(vim_util, 'get_object_property', session.vim, dsc_ref, 'podStorageDrsEntry')
    return pod_sdrs_entry.storageDrsConfig.podConfig.enabled