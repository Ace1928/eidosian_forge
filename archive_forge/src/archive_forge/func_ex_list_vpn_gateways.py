import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_list_vpn_gateways(self, account=None, domain_id=None, for_display=None, id=None, is_recursive=None, keyword=None, list_all=None, page=None, page_size=None, project_id=None, vpc_id=None):
    """
        List VPN Gateways.

        :param   account: List resources by account (must be
                          used with the domain_id parameter).
        :type    account: ``str``

        :param   domain_id: List only resources belonging
                            to the domain specified.
        :type    domain_id: ``str``

        :param   for_display: List resources by display flag (only root
                              admin is eligible to pass this parameter).
        :type    for_display: ``bool``

        :param   id: ID of the VPN Gateway.
        :type    id: ``str``

        :param   is_recursive: Defaults to False, but if true, lists all
                               resources from the parent specified by the
                               domain ID till leaves.
        :type    is_recursive: ``bool``

        :param   keyword: List by keyword.
        :type    keyword: ``str``

        :param   list_all: If set to False, list only resources belonging to
                           the command's caller; if set to True - list
                           resources that the caller is authorized to see.
                           Default value is False.
        :type    list_all: ``str``

        :param   page: Start from page.
        :type    page: ``int``

        :param   page_size: Items per page.
        :type    page_size: ``int``

        :param   project_id: List objects by project.
        :type    project_id: ``str``

        :param   vpc_id: List objects by VPC.
        :type    vpc_id: ``str``

        :rtype: ``list`` of :class:`CloudStackVpnGateway`
        """
    args = {}
    if account is not None:
        args['account'] = account
    if domain_id is not None:
        args['domainid'] = domain_id
    if for_display is not None:
        args['fordisplay'] = for_display
    if id is not None:
        args['id'] = id
    if is_recursive is not None:
        args['isrecursive'] = is_recursive
    if keyword is not None:
        args['keyword'] = keyword
    if list_all is not None:
        args['listall'] = list_all
    if page is not None:
        args['page'] = page
    if page_size is not None:
        args['pagesize'] = page_size
    if project_id is not None:
        args['projectid'] = project_id
    if vpc_id is not None:
        args['vpcid'] = vpc_id
    res = self._sync_request(command='listVpnGateways', params=args, method='GET')
    items = res.get('vpngateway', [])
    vpn_gateways = []
    extra_map = RESOURCE_EXTRA_ATTRIBUTES_MAP['vpngateway']
    for item in items:
        extra = self._get_extra_dict(item, extra_map)
        vpn_gateways.append(CloudStackVpnGateway(id=item['id'], account=item['account'], domain=item['domain'], domain_id=item['domainid'], public_ip=item['publicip'], vpc_id=item['vpcid'], driver=self, extra=extra))
    return vpn_gateways