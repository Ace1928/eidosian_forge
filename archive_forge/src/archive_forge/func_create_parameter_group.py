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
def create_parameter_group(self, name, engine='MySQL5.1', description=''):
    """
        Create a new dbparameter group for your account.

        :type name: string
        :param name: The name of the new dbparameter group

        :type engine: str
        :param engine: Name of database engine.

        :type description: string
        :param description: The description of the new dbparameter group

        :rtype: :class:`boto.rds.parametergroup.ParameterGroup`
        :return: The newly created ParameterGroup
        """
    params = {'DBParameterGroupName': name, 'DBParameterGroupFamily': engine, 'Description': description}
    return self.get_object('CreateDBParameterGroup', params, ParameterGroup)