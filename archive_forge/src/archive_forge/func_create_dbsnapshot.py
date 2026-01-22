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
def create_dbsnapshot(self, snapshot_id, dbinstance_id):
    """
        Create a new DB snapshot.

        :type snapshot_id: string
        :param snapshot_id: The identifier for the DBSnapshot

        :type dbinstance_id: string
        :param dbinstance_id: The source identifier for the RDS instance from
                              which the snapshot is created.

        :rtype: :class:`boto.rds.dbsnapshot.DBSnapshot`
        :return: The newly created DBSnapshot
        """
    params = {'DBSnapshotIdentifier': snapshot_id, 'DBInstanceIdentifier': dbinstance_id}
    return self.get_object('CreateDBSnapshot', params, DBSnapshot)