import requests
from com.vmware.vapi.std.errors_client import Unauthenticated
from vmware.vapi.vsphere.client import create_vsphere_client
from ray.autoscaler._private.vsphere.utils import Constants, singleton_client
def get_unverified_session():
    """
    vCenter provisioned internally have SSH certificates
    expired so we use unverified session. Find out what
    could be done for production.

    Get a requests session with cert verification disabled.
    Also disable the insecure warnings message.
    Note this is not recommended in production code.
    @return: a requests session with verification disabled.
    """
    session = requests.session()
    session.verify = False
    requests.packages.urllib3.disable_warnings()
    return session