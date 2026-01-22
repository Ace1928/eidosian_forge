import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def deactivate_mfa_device(self, user_name, serial_number):
    """
        Deactivates the specified MFA device and removes it from
        association with the user.

        :type user_name: string
        :param user_name: The username of the user

        :type serial_number: string
        :param serial_number: The serial number which uniquely identifies
            the MFA device.

        """
    params = {'UserName': user_name, 'SerialNumber': serial_number}
    return self.get_response('DeactivateMFADevice', params)