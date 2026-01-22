from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six import iteritems
def get_member_attributes(self):
    """ Returns a dictionary of a balancer member's attributes."""
    balancer_member_page = fetch_url(self.module, self.management_url)
    if balancer_member_page[1]['status'] != 200:
        self.module.fail_json(msg='Could not get balancer_member_page, check for connectivity! ' + balancer_member_page[1])
    else:
        try:
            soup = BeautifulSoup(balancer_member_page[0])
        except TypeError as exc:
            self.module.fail_json(msg='Cannot parse balancer_member_page HTML! ' + str(exc))
        else:
            subsoup = soup.findAll('table')[1].findAll('tr')
            keys = subsoup[0].findAll('th')
            for valuesset in subsoup[1::1]:
                if re.search(pattern=self.host, string=str(valuesset)):
                    values = valuesset.findAll('td')
                    return dict(((keys[x].string, values[x].string) for x in range(0, len(keys))))