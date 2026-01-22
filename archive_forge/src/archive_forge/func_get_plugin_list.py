import argparse
from keystoneauth1.identity.v3 import k2k
from keystoneauth1.loading import base
from osc_lib import exceptions as exc
from osc_lib.i18n import _
from osc_lib import utils
def get_plugin_list():
    """Gather plugin list and cache it"""
    global PLUGIN_LIST
    if PLUGIN_LIST is None:
        PLUGIN_LIST = base.get_available_plugin_names()
    return PLUGIN_LIST