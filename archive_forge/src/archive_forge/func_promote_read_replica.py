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
def promote_read_replica(self, id, backup_retention_period=None, preferred_backup_window=None):
    """
        Promote a Read Replica to a standalone DB Instance.

        :type id: str
        :param id: Unique identifier for the new instance.
                   Must contain 1-63 alphanumeric characters.
                   First character must be a letter.
                   May not end with a hyphen or contain two consecutive hyphens

        :type backup_retention_period: int
        :param backup_retention_period: The number of days for which automated
                                        backups are retained.  Setting this to
                                        zero disables automated backups.

        :type preferred_backup_window: str
        :param preferred_backup_window: The daily time range during which
                                        automated backups are created (if
                                        enabled).  Must be in h24:mi-hh24:mi
                                        format (UTC).

        :rtype: :class:`boto.rds.dbinstance.DBInstance`
        :return: The new db instance.
        """
    params = {'DBInstanceIdentifier': id}
    if backup_retention_period is not None:
        params['BackupRetentionPeriod'] = backup_retention_period
    if preferred_backup_window:
        params['PreferredBackupWindow'] = preferred_backup_window
    return self.get_object('PromoteReadReplica', params, DBInstance)