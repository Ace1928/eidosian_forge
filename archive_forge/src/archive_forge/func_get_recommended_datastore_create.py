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
def get_recommended_datastore_create(session, dsc_ref, config_spec, resource_pool, folder, host_ref=None):
    """Returns SDRS recommendation key for creating a VM."""
    sp_spec = vim_util.storage_placement_spec(session.vim.client.factory, dsc_ref, 'create', config_spec=config_spec, folder=folder, res_pool_ref=resource_pool, host_ref=host_ref)
    return get_recommended_datastore(session, sp_spec)