from libcloud.utils.py3 import u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.utils.misc import ReprMixin
from libcloud.common.types import LibcloudError
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
def ex_set_certificate_name(self, certificate_id, name):
    """
        Set server certificate name.

        :param certificate_id: the id of the server certificate to update
        :type certificate_id: ``str``

        :param name: the new name
        :type name: ``str``

        :return: whether updating is success
        :rtype: ``bool``
        """
    params = {'Action': 'SetServerCertificateName', 'RegionId': self.region, 'ServerCertificateId': certificate_id, 'ServerCertificateName': name}
    resp = self.connection.request(self.path, params)
    return resp.success()