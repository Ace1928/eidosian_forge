from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def is_building_updated(self, updated_site, requested_site):
    """
        Check if the building details in a site have been updated.
        Args:
            - self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            - updated_site (dict): The site details after the update.
            - requested_site (dict): The site details as requested for the update.
        Return:
            bool: True if the building details have been updated, False otherwise.
        Description:
            This method compares the building details of the updated site with the requested site.
            It checks if the name, parent_name, latitude, longitude, and address (if provided) are
            equal, indicating that the building details have been updated. Returns True if the
            details match, and False otherwise.
        """
    return updated_site['name'] == requested_site['name'] and updated_site['parentName'] == requested_site['parentName'] and self.compare_float_values(updated_site['latitude'], requested_site['latitude']) and self.compare_float_values(updated_site['longitude'], requested_site['longitude']) and ('address' in requested_site and (requested_site['address'] is None or updated_site.get('address') == requested_site['address']))