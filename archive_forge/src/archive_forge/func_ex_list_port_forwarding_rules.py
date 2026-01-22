import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_list_port_forwarding_rules(self, account=None, domain_id=None, id=None, ipaddress_id=None, is_recursive=None, keyword=None, list_all=None, network_id=None, page=None, page_size=None, project_id=None):
    """
        Lists all Port Forwarding Rules

        :param     account: List resources by account.
                            Must be used with the domainId parameter
        :type      account: ``str``

        :param     domain_id: List only resources belonging to
                                     the domain specified
        :type      domain_id: ``str``

        :param     for_display: List resources by display flag (only root
                                admin is eligible to pass this parameter).
        :type      for_display: ``bool``

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

        :param     network_id: list port forwarding rules for certain network
        :type      network_id: ``string``

        :param     page: The page to list the keypairs from
        :type      page: ``int``

        :param     page_size: The number of results per page
        :type      page_size: ``int``

        :param     project_id: list objects by project
        :type      project_id: ``str``

        :rtype: ``list`` of :class:`CloudStackPortForwardingRule`
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
    if network_id is not None:
        args['networkid'] = network_id
    if page is not None:
        args['page'] = page
    if page_size is not None:
        args['pagesize'] = page_size
    if project_id is not None:
        args['projectid'] = project_id
    rules = []
    result = self._sync_request(command='listPortForwardingRules', params=args, method='GET')
    if result != {}:
        public_ips = self.ex_list_public_ips()
        nodes = self.list_nodes()
        for rule in result['portforwardingrule']:
            node = [n for n in nodes if n.id == str(rule['virtualmachineid'])]
            addr = [a for a in public_ips if a.address == rule['ipaddress']]
            rules.append(CloudStackPortForwardingRule(node[0], rule['id'], addr[0], rule['protocol'], rule['publicport'], rule['privateport'], rule['publicendport'], rule['privateendport']))
    return rules