import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_list_ip_forwarding_rules(self, account=None, domain_id=None, id=None, ipaddress_id=None, is_recursive=None, keyword=None, list_all=None, page=None, page_size=None, project_id=None, virtualmachine_id=None):
    """
        Lists all NAT/firewall forwarding rules

        :param     account: List resources by account.
                            Must be used with the domainId parameter
        :type      account: ``str``

        :param     domain_id: List only resources belonging to
                                     the domain specified
        :type      domain_id: ``str``

        :param     id: Lists rule with the specified ID
        :type      id: ``str``

        :param     ipaddress_id: list the rule belonging to
                                this public ip address
        :type      ipaddress_id: ``str``

        :param     is_recursive: Defaults to false, but if true,
                                lists all resources from
                                the parent specified by the
                                domainId till leaves.
        :type      is_recursive: ``bool``

        :param     keyword: List by keyword
        :type      keyword: ``str``

        :param     list_all: If set to false, list only resources
                            belonging to the command's caller;
                            if set to true - list resources that
                            the caller is authorized to see.
                            Default value is false
        :type      list_all: ``bool``

        :param     page: The page to list the keypairs from
        :type      page: ``int``

        :param     page_size: The number of results per page
        :type      page_size: ``int``

        :param     project_id: list objects by project
        :type      project_id: ``str``

        :param     virtualmachine_id: Lists all rules applied to
                                     the specified Vm
        :type      virtualmachine_id: ``str``

        :rtype: ``list`` of :class:`CloudStackIPForwardingRule`
        """
    args = {}
    if account is not None:
        args['account'] = account
    if domain_id is not None:
        args['domainid'] = domain_id
    if id is not None:
        args['id'] = id
    if ipaddress_id is not None:
        args['ipaddressid'] = ipaddress_id
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
    if virtualmachine_id is not None:
        args['virtualmachineid'] = virtualmachine_id
    result = self._sync_request(command='listIpForwardingRules', params=args, method='GET')
    rules = []
    if result != {}:
        public_ips = self.ex_list_public_ips()
        nodes = self.list_nodes()
        for rule in result['ipforwardingrule']:
            node = [n for n in nodes if n.id == str(rule['virtualmachineid'])]
            addr = [a for a in public_ips if a.address == rule['ipaddress']]
            rules.append(CloudStackIPForwardingRule(node[0], rule['id'], addr[0], rule['protocol'], rule['startport'], rule['endport']))
    return rules