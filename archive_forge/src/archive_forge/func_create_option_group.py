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
def create_option_group(self, name, engine_name, major_engine_version, description=None):
    """
        Create a new option group for your account.
        This will create the option group within the region you
        are currently connected to.

        :type name: string
        :param name: The name of the new option group

        :type engine_name: string
        :param engine_name: Specifies the name of the engine that this option
                            group should be associated with.

        :type major_engine_version: string
        :param major_engine_version: Specifies the major version of the engine
                                     that this option group should be
                                     associated with.

        :type description: string
        :param description: The description of the new option group

        :rtype: :class:`boto.rds.optiongroup.OptionGroup`
        :return: The newly created OptionGroup
        """
    params = {'OptionGroupName': name, 'EngineName': engine_name, 'MajorEngineVersion': major_engine_version, 'OptionGroupDescription': description}
    group = self.get_object('CreateOptionGroup', params, OptionGroup)
    group.name = name
    group.engine_name = engine_name
    group.major_engine_version = major_engine_version
    group.description = description
    return group