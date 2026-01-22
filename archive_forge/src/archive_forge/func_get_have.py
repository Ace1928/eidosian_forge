from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_have(self, config):
    """
        Get the site details from Cisco Catalyst Center
        Parameters:
          - self (object): An instance of a class used for interacting with Cisco Catalyst Center.
          - config (dict): A dictionary containing the configuration details.
        Returns:
          - self (object): An instance of a class used for interacting with  Cisco Catalyst Center.
        Description:
            This method queries Cisco Catalyst Center to check if a specified site
            exists. If the site exists, it retrieves details about the current
            site, including the site ID and other relevant information. The
            results are stored in the 'have' attribute for later reference.
        """
    site_exists = False
    current_site = None
    have = {}
    site_exists, current_site = self.site_exists()
    self.log('Current Site details (have): {0}'.format(str(current_site)), 'DEBUG')
    if site_exists:
        have['site_id'] = current_site.get('siteId')
        have['site_exists'] = site_exists
        have['current_site'] = current_site
    self.have = have
    self.log('Current State (have): {0}'.format(str(self.have)), 'INFO')
    return self