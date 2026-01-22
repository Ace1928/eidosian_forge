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
def revoke_dbsecurity_group(self, group_name, ec2_security_group_name=None, ec2_security_group_owner_id=None, cidr_ip=None):
    """
        Remove an existing rule from an existing security group.
        You need to pass in either ec2_security_group_name and
        ec2_security_group_owner_id OR a CIDR block.

        :type group_name: string
        :param group_name: The name of the security group you are removing
                           the rule from.

        :type ec2_security_group_name: string
        :param ec2_security_group_name: The name of the EC2 security group
                                        from which you are removing access.

        :type ec2_security_group_owner_id: string
        :param ec2_security_group_owner_id: The ID of the owner of the EC2
                                            security from which you are
                                            removing access.

        :type cidr_ip: string
        :param cidr_ip: The CIDR block from which you are removing access.
                        See http://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing

        :rtype: bool
        :return: True if successful.
        """
    params = {'DBSecurityGroupName': group_name}
    if ec2_security_group_name:
        params['EC2SecurityGroupName'] = ec2_security_group_name
    if ec2_security_group_owner_id:
        params['EC2SecurityGroupOwnerId'] = ec2_security_group_owner_id
    if cidr_ip:
        params['CIDRIP'] = cidr_ip
    return self.get_object('RevokeDBSecurityGroupIngress', params, DBSecurityGroup)