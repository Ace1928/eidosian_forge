import types
import boto
import boto.utils
from boto.ec2.regioninfo import RegionInfo
from boto.emr.emrobject import AddInstanceGroupsResponse, BootstrapActionList, \
from boto.emr.step import JarStep
from boto.connection import AWSQueryConnection
from boto.exception import EmrResponseError
from boto.compat import six
def terminate_jobflow(self, jobflow_id):
    """
        Terminate an Elastic MapReduce job flow

        :type jobflow_id: str
        :param jobflow_id: A jobflow id
        """
    self.terminate_jobflows([jobflow_id])