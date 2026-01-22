import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudhsm import exceptions
def modify_hapg(self, hapg_arn, label=None, partition_serial_list=None):
    """
        Modifies an existing high-availability partition group.

        :type hapg_arn: string
        :param hapg_arn: The ARN of the high-availability partition group to
            modify.

        :type label: string
        :param label: The new label for the high-availability partition group.

        :type partition_serial_list: list
        :param partition_serial_list: The list of partition serial numbers to
            make members of the high-availability partition group.

        """
    params = {'HapgArn': hapg_arn}
    if label is not None:
        params['Label'] = label
    if partition_serial_list is not None:
        params['PartitionSerialList'] = partition_serial_list
    return self.make_request(action='ModifyHapg', body=json.dumps(params))