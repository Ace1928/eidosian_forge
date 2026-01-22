import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.redshift import exceptions
def create_cluster_subnet_group(self, cluster_subnet_group_name, description, subnet_ids):
    """
        Creates a new Amazon Redshift subnet group. You must provide a
        list of one or more subnets in your existing Amazon Virtual
        Private Cloud (Amazon VPC) when creating Amazon Redshift
        subnet group.

        For information about subnet groups, go to `Amazon Redshift
        Cluster Subnet Groups`_ in the Amazon Redshift Management
        Guide .

        :type cluster_subnet_group_name: string
        :param cluster_subnet_group_name: The name for the subnet group. Amazon
            Redshift stores the value as a lowercase string.
        Constraints:


        + Must contain no more than 255 alphanumeric characters or hyphens.
        + Must not be "Default".
        + Must be unique for all subnet groups that are created by your AWS
              account.


        Example: `examplesubnetgroup`

        :type description: string
        :param description: A description for the subnet group.

        :type subnet_ids: list
        :param subnet_ids: An array of VPC subnet IDs. A maximum of 20 subnets
            can be modified in a single request.

        """
    params = {'ClusterSubnetGroupName': cluster_subnet_group_name, 'Description': description}
    self.build_list_params(params, subnet_ids, 'SubnetIds.member')
    return self._make_request(action='CreateClusterSubnetGroup', verb='POST', path='/', params=params)