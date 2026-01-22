import ipaddress
import time
import warnings
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
from openstack import proxy
from openstack import utils
from openstack import warnings as os_warnings
def delete_unattached_floating_ips(self, retry=1):
    """Safely delete unattached floating ips.

        If the cloud can safely purge any unattached floating ips without
        race conditions, do so.

        Safely here means a specific thing. It means that you are not running
        this while another process that might do a two step create/attach
        is running. You can safely run this  method while another process
        is creating servers and attaching floating IPs to them if either that
        process is using add_auto_ip from shade, or is creating the floating
        IPs by passing in a server to the create_floating_ip call.

        :param retry: number of times to retry. Optional, defaults to 1,
                      which is in addition to the initial delete call.
                      A value of 0 will also cause no checking of results to
                      occur.

        :returns: Number of Floating IPs deleted, False if none
        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        """
    processed = []
    if self._use_neutron_floating():
        for ip in self.list_floating_ips():
            if not bool(ip.port_id):
                processed.append(self.delete_floating_ip(floating_ip_id=ip['id'], retry=retry))
    return len(processed) if all(processed) else False