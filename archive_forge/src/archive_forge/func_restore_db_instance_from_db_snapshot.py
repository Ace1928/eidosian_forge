import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def restore_db_instance_from_db_snapshot(self, db_instance_identifier, db_snapshot_identifier, db_instance_class=None, port=None, availability_zone=None, db_subnet_group_name=None, multi_az=None, publicly_accessible=None, auto_minor_version_upgrade=None, license_model=None, db_name=None, engine=None, iops=None, option_group_name=None, tags=None):
    """
        Creates a new DB instance from a DB snapshot. The target
        database is created from the source database restore point
        with the same configuration as the original source database,
        except that the new RDS instance is created with the default
        security group.

        :type db_instance_identifier: string
        :param db_instance_identifier:
        The identifier for the DB snapshot to restore from.

        Constraints:


        + Must contain from 1 to 63 alphanumeric characters or hyphens
        + First character must be a letter
        + Cannot end with a hyphen or contain two consecutive hyphens

        :type db_snapshot_identifier: string
        :param db_snapshot_identifier: Name of the DB instance to create from
            the DB snapshot. This parameter isn't case sensitive.
        Constraints:


        + Must contain from 1 to 255 alphanumeric characters or hyphens
        + First character must be a letter
        + Cannot end with a hyphen or contain two consecutive hyphens


        Example: `my-snapshot-id`

        :type db_instance_class: string
        :param db_instance_class: The compute and memory capacity of the Amazon
            RDS DB instance.
        Valid Values: `db.t1.micro | db.m1.small | db.m1.medium | db.m1.large |
            db.m1.xlarge | db.m2.2xlarge | db.m2.4xlarge`

        :type port: integer
        :param port: The port number on which the database accepts connections.
        Default: The same port as the original DB instance

        Constraints: Value must be `1150-65535`

        :type availability_zone: string
        :param availability_zone: The EC2 Availability Zone that the database
            instance will be created in.
        Default: A random, system-chosen Availability Zone.

        Constraint: You cannot specify the AvailabilityZone parameter if the
            MultiAZ parameter is set to `True`.

        Example: `us-east-1a`

        :type db_subnet_group_name: string
        :param db_subnet_group_name: The DB subnet group name to use for the
            new instance.

        :type multi_az: boolean
        :param multi_az: Specifies if the DB instance is a Multi-AZ deployment.
        Constraint: You cannot specify the AvailabilityZone parameter if the
            MultiAZ parameter is set to `True`.

        :type publicly_accessible: boolean
        :param publicly_accessible: Specifies the accessibility options for the
            DB instance. A value of true specifies an Internet-facing instance
            with a publicly resolvable DNS name, which resolves to a public IP
            address. A value of false specifies an internal instance with a DNS
            name that resolves to a private IP address.
        Default: The default behavior varies depending on whether a VPC has
            been requested or not. The following list shows the default
            behavior in each case.


        + **Default VPC:**true
        + **VPC:**false


        If no DB subnet group has been specified as part of the request and the
            PubliclyAccessible value has not been set, the DB instance will be
            publicly accessible. If a specific DB subnet group has been
            specified as part of the request and the PubliclyAccessible value
            has not been set, the DB instance will be private.

        :type auto_minor_version_upgrade: boolean
        :param auto_minor_version_upgrade: Indicates that minor version
            upgrades will be applied automatically to the DB instance during
            the maintenance window.

        :type license_model: string
        :param license_model: License model information for the restored DB
            instance.
        Default: Same as source.

        Valid values: `license-included` | `bring-your-own-license` | `general-
            public-license`

        :type db_name: string
        :param db_name:
        The database name for the restored DB instance.


        This parameter doesn't apply to the MySQL engine.

        :type engine: string
        :param engine: The database engine to use for the new instance.
        Default: The same as source

        Constraint: Must be compatible with the engine of the source

        Example: `oracle-ee`

        :type iops: integer
        :param iops: Specifies the amount of provisioned IOPS for the DB
            instance, expressed in I/O operations per second. If this parameter
            is not specified, the IOPS value will be taken from the backup. If
            this parameter is set to 0, the new instance will be converted to a
            non-PIOPS instance, which will take additional time, though your DB
            instance will be available for connections before the conversion
            starts.
        Constraints: Must be an integer greater than 1000.

        :type option_group_name: string
        :param option_group_name: The name of the option group to be used for
            the restored DB instance.
        Permanent options, such as the TDE option for Oracle Advanced Security
            TDE, cannot be removed from an option group, and that option group
            cannot be removed from a DB instance once it is associated with a
            DB instance

        :type tags: list
        :param tags: A list of tags. Tags must be passed as tuples in the form
            [('key1', 'valueForKey1'), ('key2', 'valueForKey2')]

        """
    params = {'DBInstanceIdentifier': db_instance_identifier, 'DBSnapshotIdentifier': db_snapshot_identifier}
    if db_instance_class is not None:
        params['DBInstanceClass'] = db_instance_class
    if port is not None:
        params['Port'] = port
    if availability_zone is not None:
        params['AvailabilityZone'] = availability_zone
    if db_subnet_group_name is not None:
        params['DBSubnetGroupName'] = db_subnet_group_name
    if multi_az is not None:
        params['MultiAZ'] = str(multi_az).lower()
    if publicly_accessible is not None:
        params['PubliclyAccessible'] = str(publicly_accessible).lower()
    if auto_minor_version_upgrade is not None:
        params['AutoMinorVersionUpgrade'] = str(auto_minor_version_upgrade).lower()
    if license_model is not None:
        params['LicenseModel'] = license_model
    if db_name is not None:
        params['DBName'] = db_name
    if engine is not None:
        params['Engine'] = engine
    if iops is not None:
        params['Iops'] = iops
    if option_group_name is not None:
        params['OptionGroupName'] = option_group_name
    if tags is not None:
        self.build_complex_list_params(params, tags, 'Tags.member', ('Key', 'Value'))
    return self._make_request(action='RestoreDBInstanceFromDBSnapshot', verb='POST', path='/', params=params)