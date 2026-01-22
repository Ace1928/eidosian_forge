import logging
import os
import urllib.parse as urlparse
import urllib.request as urllib
from oslo_vmware import service
from oslo_vmware import vim_util
def filter_datastores_by_hubs(hubs, datastores):
    """Get filtered subset of datastores corresponding to the given hub list.

    :param hubs: list of PbmPlacementHub morefs
    :param datastores: all candidate datastores
    :returns: subset of datastores corresponding to the given hub list
    """
    filtered_dss = []
    hub_ids = [hub.hubId for hub in hubs]
    for ds in datastores:
        if vim_util.get_moref_value(ds) in hub_ids:
            filtered_dss.append(ds)
    return filtered_dss