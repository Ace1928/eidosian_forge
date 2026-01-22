import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.redshift import exceptions
def delete_cluster_subnet_group(self, cluster_subnet_group_name):
    """
        Deletes the specified cluster subnet group.

        :type cluster_subnet_group_name: string
        :param cluster_subnet_group_name: The name of the cluster subnet group
            name to be deleted.

        """
    params = {'ClusterSubnetGroupName': cluster_subnet_group_name}
    return self._make_request(action='DeleteClusterSubnetGroup', verb='POST', path='/', params=params)