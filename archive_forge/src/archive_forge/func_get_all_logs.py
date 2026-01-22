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
def get_all_logs(self, dbinstance_id, max_records=None, marker=None, file_size=None, filename_contains=None, file_last_written=None):
    """
        Get all log files

        :type instance_id: str
        :param instance_id: The identifier of a DBInstance.

        :type max_records: int
        :param max_records: Number of log file names to return.

        :type marker: str
        :param marker: The marker provided by a previous request.

        :file_size: int
        :param file_size: Filter results to files large than this size in bytes.

        :filename_contains: str
        :param filename_contains: Filter results to files with filename containing this string

        :file_last_written: int
        :param file_last_written: Filter results to files written after this time (POSIX timestamp)

        :rtype: list
        :return: A list of :class:`boto.rds.logfile.LogFile`
        """
    params = {'DBInstanceIdentifier': dbinstance_id}
    if file_size:
        params['FileSize'] = file_size
    if filename_contains:
        params['FilenameContains'] = filename_contains
    if file_last_written:
        params['FileLastWritten'] = file_last_written
    if marker:
        params['Marker'] = marker
    if max_records:
        params['MaxRecords'] = max_records
    return self.get_list('DescribeDBLogFiles', params, [('DescribeDBLogFilesDetails', LogFile)])