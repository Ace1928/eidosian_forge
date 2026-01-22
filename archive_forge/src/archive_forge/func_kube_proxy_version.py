from pprint import pformat
from six import iteritems
import re
@kube_proxy_version.setter
def kube_proxy_version(self, kube_proxy_version):
    """
        Sets the kube_proxy_version of this V1NodeSystemInfo.
        KubeProxy Version reported by the node.

        :param kube_proxy_version: The kube_proxy_version of this
        V1NodeSystemInfo.
        :type: str
        """
    if kube_proxy_version is None:
        raise ValueError('Invalid value for `kube_proxy_version`, must not be `None`')
    self._kube_proxy_version = kube_proxy_version