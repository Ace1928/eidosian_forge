from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_udf_id(self, field_name):
    """
        Get the ID of a Global User Defined Field in Cisco Catalyst Center based on its name.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Cisco Catalyst Center.
            field_name (str): The name of the Global User Defined Field.
        Returns:
            str: The ID of the Global User Defined Field.
        Description:
            The function sends a request to Cisco Catalyst Center to retrieve all Global User Defined Fields
            with the specified name and extracts the ID of the first matching field.If successful, it returns
            the ID else returns None.
        """
    try:
        udf_id = None
        response = self.dnac._exec(family='devices', function='get_all_user_defined_fields', params={'name': field_name})
        self.log("Received API response from 'get_all_user_defined_fields': {0}".format(str(response)), 'DEBUG')
        udf = response.get('response')
        if udf:
            udf_id = udf[0].get('id')
    except Exception as e:
        error_message = 'Exception occurred while getting Global User Defined Fields(UDF) ID from Cisco Catalyst Center: {0}'.format(str(e))
        self.log(error_message, 'ERROR')
    return udf_id