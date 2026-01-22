import types
import boto
import boto.utils
from boto.ec2.regioninfo import RegionInfo
from boto.emr.emrobject import AddInstanceGroupsResponse, BootstrapActionList, \
from boto.emr.step import JarStep
from boto.connection import AWSQueryConnection
from boto.exception import EmrResponseError
from boto.compat import six
def list_bootstrap_actions(self, cluster_id, marker=None):
    """
        Get a list of bootstrap actions for an Elastic MapReduce cluster

        :type cluster_id: str
        :param cluster_id: The cluster id of interest
        :type marker: str
        :param marker: Pagination marker
        """
    params = {'ClusterId': cluster_id}
    if marker:
        params['Marker'] = marker
    return self.get_object('ListBootstrapActions', params, BootstrapActionList)