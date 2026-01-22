import urllib
from boto.connection import AWSQueryConnection
from boto.rds.dbinstance import DBInstance
from boto.rds.dbsecuritygroup import DBSecurityGroup
from boto.rds.optiongroup  import OptionGroup, OptionGroupOption
from boto.rds.parametergroup import ParameterGroup
from boto.rds.dbsnapshot import DBSnapshot
from boto.rds.event import Event
from boto.rds.regioninfo import RDSRegionInfo
from boto.rds.dbsubnetgroup import DBSubnetGroup
from boto.rds.vpcsecuritygroupmembership import VPCSecurityGroupMembership
from boto.regioninfo import get_regions
from boto.regioninfo import connect
from boto.rds.logfile import LogFile, LogFileObject
def get_all_dbparameter_groups(self, groupname=None, max_records=None, marker=None):
    """
        Get all parameter groups associated with your account in a region.

        :type groupname: str
        :param groupname: The name of the DBParameter group to retrieve.
                          If not provided, all DBParameter groups will be returned.

        :type max_records: int
        :param max_records: The maximum number of records to be returned.
                            If more results are available, a MoreToken will
                            be returned in the response that can be used to
                            retrieve additional records.  Default is 100.

        :type marker: str
        :param marker: The marker provided by a previous request.

        :rtype: list
        :return: A list of :class:`boto.ec2.parametergroup.ParameterGroup`
        """
    params = {}
    if groupname:
        params['DBParameterGroupName'] = groupname
    if max_records:
        params['MaxRecords'] = max_records
    if marker:
        params['Marker'] = marker
    return self.get_list('DescribeDBParameterGroups', params, [('DBParameterGroup', ParameterGroup)])