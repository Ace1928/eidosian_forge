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
def restore_dbinstance_from_point_in_time(self, source_instance_id, target_instance_id, use_latest=False, restore_time=None, dbinstance_class=None, port=None, availability_zone=None, db_subnet_group_name=None):
    """
        Create a new DBInstance from a point in time.

        :type source_instance_id: string
        :param source_instance_id: The identifier for the source DBInstance.

        :type target_instance_id: string
        :param target_instance_id: The identifier of the new DBInstance.

        :type use_latest: bool
        :param use_latest: If True, the latest snapshot availabile will
                           be used.

        :type restore_time: datetime
        :param restore_time: The date and time to restore from.  Only
                             used if use_latest is False.

        :type instance_class: str
        :param instance_class: The compute and memory capacity of the
                               DBInstance.  Valid values are:
                               db.m1.small | db.m1.large | db.m1.xlarge |
                               db.m2.2xlarge | db.m2.4xlarge

        :type port: int
        :param port: Port number on which database accepts connections.
                     Valid values [1115-65535].  Defaults to 3306.

        :type availability_zone: str
        :param availability_zone: Name of the availability zone to place
                                  DBInstance into.

        :type db_subnet_group_name: str
        :param db_subnet_group_name: A DB Subnet Group to associate with this DB Instance.
                                     If there is no DB Subnet Group, then it is a non-VPC DB
                                     instance.

        :rtype: :class:`boto.rds.dbinstance.DBInstance`
        :return: The newly created DBInstance
        """
    params = {'SourceDBInstanceIdentifier': source_instance_id, 'TargetDBInstanceIdentifier': target_instance_id}
    if use_latest:
        params['UseLatestRestorableTime'] = 'true'
    elif restore_time:
        params['RestoreTime'] = restore_time.isoformat()
    if dbinstance_class:
        params['DBInstanceClass'] = dbinstance_class
    if port:
        params['Port'] = port
    if availability_zone:
        params['AvailabilityZone'] = availability_zone
    if db_subnet_group_name is not None:
        params['DBSubnetGroupName'] = db_subnet_group_name
    return self.get_object('RestoreDBInstanceToPointInTime', params, DBInstance)