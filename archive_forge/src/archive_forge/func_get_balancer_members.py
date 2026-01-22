from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six import iteritems
def get_balancer_members(self):
    """ Returns members of the balancer as a generator object for later iteration."""
    try:
        soup = BeautifulSoup(self.page)
    except TypeError:
        self.module.fail_json(msg='Cannot parse balancer page HTML! ' + str(self.page))
    else:
        for element in soup.findAll('a')[1::1]:
            balancer_member_suffix = str(element.get('href'))
            if not balancer_member_suffix:
                self.module.fail_json(msg="Argument 'balancer_member_suffix' is empty!")
            else:
                yield BalancerMember(str(self.base_url + balancer_member_suffix), str(self.url), self.module)