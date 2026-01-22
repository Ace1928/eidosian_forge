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
def describe_option_groups(self, name=None, engine_name=None, major_engine_version=None, max_records=100, marker=None):
    """
        Describes the available option groups.

        :type name: str
        :param name: The name of the option group to describe. Cannot be
                     supplied together with engine_name or major_engine_version.

        :type engine_name: str
        :param engine_name: Filters the list of option groups to only include
                            groups associated with a specific database engine.

        :type major_engine_version: datetime
        :param major_engine_version: Filters the list of option groups to only
                                     include groups associated with a specific
                                     database engine version. If specified, then
                                     engine_name must also be specified.

        :type max_records: int
        :param max_records: The maximum number of records to be returned.
                            If more results are available, a MoreToken will
                            be returned in the response that can be used to
                            retrieve additional records.  Default is 100.

        :type marker: str
        :param marker: The marker provided by a previous request.

        :rtype: list
        :return: A list of class:`boto.rds.optiongroup.OptionGroup`
        """
    params = {}
    if name:
        params['OptionGroupName'] = name
    elif engine_name and major_engine_version:
        params['EngineName'] = engine_name
        params['MajorEngineVersion'] = major_engine_version
    if max_records:
        params['MaxRecords'] = int(max_records)
    if marker:
        params['Marker'] = marker
    return self.get_list('DescribeOptionGroups', params, [('OptionGroup', OptionGroup)])