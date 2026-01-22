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
def reset_parameter_group(self, name, reset_all_params=False, parameters=None):
    """
        Resets some or all of the parameters of a ParameterGroup to the
        default value

        :type key_name: string
        :param key_name: The name of the ParameterGroup to reset

        :type parameters: list of :class:`boto.rds.parametergroup.Parameter`
        :param parameters: The parameters to reset.  If not supplied,
                           all parameters will be reset.
        """
    params = {'DBParameterGroupName': name}
    if reset_all_params:
        params['ResetAllParameters'] = 'true'
    else:
        params['ResetAllParameters'] = 'false'
        for i in range(0, len(parameters)):
            parameter = parameters[i]
            parameter.merge(params, i + 1)
    return self.get_status('ResetDBParameterGroup', params)