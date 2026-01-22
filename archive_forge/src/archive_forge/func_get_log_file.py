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
def get_log_file(self, dbinstance_id, log_file_name, marker=None, number_of_lines=None, max_records=None):
    """
        Download a log file from RDS

        :type instance_id: str
        :param instance_id: The identifier of a DBInstance.

        :type log_file_name: str
        :param log_file_name: The name of the log file to retrieve

        :type marker: str
        :param marker: A marker returned from a previous call to this method, or 0 to indicate the start of file. If
                       no marker is specified, this will fetch log lines from the end of file instead.

        :type number_of_lines: int
        :param marker: The maximium number of lines to be returned.
        """
    params = {'DBInstanceIdentifier': dbinstance_id, 'LogFileName': log_file_name}
    if marker:
        params['Marker'] = marker
    if number_of_lines:
        params['NumberOfLines'] = number_of_lines
    if max_records:
        params['MaxRecords'] = max_records
    logfile = self.get_object('DownloadDBLogFilePortion', params, LogFileObject)
    if logfile:
        logfile.log_filename = log_file_name
        logfile.dbinstance_id = dbinstance_id
    return logfile