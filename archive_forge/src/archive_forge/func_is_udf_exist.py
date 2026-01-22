from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def is_udf_exist(self, field_name):
    """
        Check if a Global User Defined Field exists in Cisco Catalyst Center based on its name.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            field_name (str): The name of the Global User Defined Field.
        Returns:
            bool: True if the Global User Defined Field exists, False otherwise.
        Description:
            The function sends a request to Cisco Catalyst Center to retrieve all Global User Defined Fields
            with the specified name. If matching field is found, the function returns True, indicating that
            the field exists else returns False.
        """
    response = self.dnac._exec(family='devices', function='get_all_user_defined_fields', params={'name': field_name})
    self.log("Received API response from 'get_all_user_defined_fields': {0}".format(str(response)), 'DEBUG')
    udf = response.get('response')
    if len(udf) == 1:
        return True
    message = "Global User Defined Field with name '{0}' doesnot exist in Cisco Catalyst Center".format(field_name)
    self.log(message, 'INFO')
    return False