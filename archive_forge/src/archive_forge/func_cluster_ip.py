from pprint import pformat
from six import iteritems
import re
@cluster_ip.setter
def cluster_ip(self, cluster_ip):
    """
        Sets the cluster_ip of this V1ServiceSpec.
        clusterIP is the IP address of the service and is usually assigned
        randomly by the master. If an address is specified manually and is not
        in use by others, it will be allocated to the service; otherwise,
        creation of the service will fail. This field can not be changed through
        updates. Valid values are "None", empty string (""), or a valid IP
        address. "None" can be specified for headless services when proxying
        is not required. Only applies to types ClusterIP, NodePort, and
        LoadBalancer. Ignored if type is ExternalName. More info:
        https://kubernetes.io/docs/concepts/services-networking/service/#virtual-ips-and-service-proxies

        :param cluster_ip: The cluster_ip of this V1ServiceSpec.
        :type: str
        """
    self._cluster_ip = cluster_ip