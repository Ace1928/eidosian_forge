import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.redshift import exceptions
def restore_from_cluster_snapshot(self, cluster_identifier, snapshot_identifier, snapshot_cluster_identifier=None, port=None, availability_zone=None, allow_version_upgrade=None, cluster_subnet_group_name=None, publicly_accessible=None, owner_account=None, hsm_client_certificate_identifier=None, hsm_configuration_identifier=None, elastic_ip=None, cluster_parameter_group_name=None, cluster_security_groups=None, vpc_security_group_ids=None, preferred_maintenance_window=None, automated_snapshot_retention_period=None):
    """
        Creates a new cluster from a snapshot. Amazon Redshift creates
        the resulting cluster with the same configuration as the
        original cluster from which the snapshot was created, except
        that the new cluster is created with the default cluster
        security and parameter group. After Amazon Redshift creates
        the cluster you can use the ModifyCluster API to associate a
        different security group and different parameter group with
        the restored cluster.

        If you restore a cluster into a VPC, you must provide a
        cluster subnet group where you want the cluster restored.

        For more information about working with snapshots, go to
        `Amazon Redshift Snapshots`_ in the Amazon Redshift Management
        Guide .

        :type cluster_identifier: string
        :param cluster_identifier: The identifier of the cluster that will be
            created from restoring the snapshot.

        Constraints:


        + Must contain from 1 to 63 alphanumeric characters or hyphens.
        + Alphabetic characters must be lowercase.
        + First character must be a letter.
        + Cannot end with a hyphen or contain two consecutive hyphens.
        + Must be unique for all clusters within an AWS account.

        :type snapshot_identifier: string
        :param snapshot_identifier: The name of the snapshot from which to
            create the new cluster. This parameter isn't case sensitive.
        Example: `my-snapshot-id`

        :type snapshot_cluster_identifier: string
        :param snapshot_cluster_identifier: The name of the cluster the source
            snapshot was created from. This parameter is required if your IAM
            user has a policy containing a snapshot resource element that
            specifies anything other than * for the cluster name.

        :type port: integer
        :param port: The port number on which the cluster accepts connections.
        Default: The same port as the original cluster.

        Constraints: Must be between `1115` and `65535`.

        :type availability_zone: string
        :param availability_zone: The Amazon EC2 Availability Zone in which to
            restore the cluster.
        Default: A random, system-chosen Availability Zone.

        Example: `us-east-1a`

        :type allow_version_upgrade: boolean
        :param allow_version_upgrade: If `True`, upgrades can be applied during
            the maintenance window to the Amazon Redshift engine that is
            running on the cluster.
        Default: `True`

        :type cluster_subnet_group_name: string
        :param cluster_subnet_group_name: The name of the subnet group where
            you want to cluster restored.
        A snapshot of cluster in VPC can be restored only in VPC. Therefore,
            you must provide subnet group name where you want the cluster
            restored.

        :type publicly_accessible: boolean
        :param publicly_accessible: If `True`, the cluster can be accessed from
            a public network.

        :type owner_account: string
        :param owner_account: The AWS customer account used to create or copy
            the snapshot. Required if you are restoring a snapshot you do not
            own, optional if you own the snapshot.

        :type hsm_client_certificate_identifier: string
        :param hsm_client_certificate_identifier: Specifies the name of the HSM
            client certificate the Amazon Redshift cluster uses to retrieve the
            data encryption keys stored in an HSM.

        :type hsm_configuration_identifier: string
        :param hsm_configuration_identifier: Specifies the name of the HSM
            configuration that contains the information the Amazon Redshift
            cluster can use to retrieve and store keys in an HSM.

        :type elastic_ip: string
        :param elastic_ip: The elastic IP (EIP) address for the cluster.

        :type cluster_parameter_group_name: string
        :param cluster_parameter_group_name:
        The name of the parameter group to be associated with this cluster.

        Default: The default Amazon Redshift cluster parameter group. For
            information about the default parameter group, go to `Working with
            Amazon Redshift Parameter Groups`_.

        Constraints:


        + Must be 1 to 255 alphanumeric characters or hyphens.
        + First character must be a letter.
        + Cannot end with a hyphen or contain two consecutive hyphens.

        :type cluster_security_groups: list
        :param cluster_security_groups: A list of security groups to be
            associated with this cluster.
        Default: The default cluster security group for Amazon Redshift.

        Cluster security groups only apply to clusters outside of VPCs.

        :type vpc_security_group_ids: list
        :param vpc_security_group_ids: A list of Virtual Private Cloud (VPC)
            security groups to be associated with the cluster.
        Default: The default VPC security group is associated with the cluster.

        VPC security groups only apply to clusters in VPCs.

        :type preferred_maintenance_window: string
        :param preferred_maintenance_window: The weekly time range (in UTC)
            during which automated cluster maintenance can occur.
        Format: `ddd:hh24:mi-ddd:hh24:mi`

        Default: The value selected for the cluster from which the snapshot was
            taken. The following list shows the time blocks for each region
            from which the default maintenance windows are assigned.


        + **US-East (Northern Virginia) Region:** 03:00-11:00 UTC
        + **US-West (Oregon) Region** 06:00-14:00 UTC
        + **EU (Ireland) Region** 22:00-06:00 UTC
        + **Asia Pacific (Singapore) Region** 14:00-22:00 UTC
        + **Asia Pacific (Sydney) Region** 12:00-20:00 UTC
        + **Asia Pacific (Tokyo) Region** 17:00-03:00 UTC


        Valid Days: Mon | Tue | Wed | Thu | Fri | Sat | Sun

        Constraints: Minimum 30-minute window.

        :type automated_snapshot_retention_period: integer
        :param automated_snapshot_retention_period: The number of days that
            automated snapshots are retained. If the value is 0, automated
            snapshots are disabled. Even if automated snapshots are disabled,
            you can still create manual snapshots when you want with
            CreateClusterSnapshot.
        Default: The value selected for the cluster from which the snapshot was
            taken.

        Constraints: Must be a value from 0 to 35.

        """
    params = {'ClusterIdentifier': cluster_identifier, 'SnapshotIdentifier': snapshot_identifier}
    if snapshot_cluster_identifier is not None:
        params['SnapshotClusterIdentifier'] = snapshot_cluster_identifier
    if port is not None:
        params['Port'] = port
    if availability_zone is not None:
        params['AvailabilityZone'] = availability_zone
    if allow_version_upgrade is not None:
        params['AllowVersionUpgrade'] = str(allow_version_upgrade).lower()
    if cluster_subnet_group_name is not None:
        params['ClusterSubnetGroupName'] = cluster_subnet_group_name
    if publicly_accessible is not None:
        params['PubliclyAccessible'] = str(publicly_accessible).lower()
    if owner_account is not None:
        params['OwnerAccount'] = owner_account
    if hsm_client_certificate_identifier is not None:
        params['HsmClientCertificateIdentifier'] = hsm_client_certificate_identifier
    if hsm_configuration_identifier is not None:
        params['HsmConfigurationIdentifier'] = hsm_configuration_identifier
    if elastic_ip is not None:
        params['ElasticIp'] = elastic_ip
    if cluster_parameter_group_name is not None:
        params['ClusterParameterGroupName'] = cluster_parameter_group_name
    if cluster_security_groups is not None:
        self.build_list_params(params, cluster_security_groups, 'ClusterSecurityGroups.member')
    if vpc_security_group_ids is not None:
        self.build_list_params(params, vpc_security_group_ids, 'VpcSecurityGroupIds.member')
    if preferred_maintenance_window is not None:
        params['PreferredMaintenanceWindow'] = preferred_maintenance_window
    if automated_snapshot_retention_period is not None:
        params['AutomatedSnapshotRetentionPeriod'] = automated_snapshot_retention_period
    return self._make_request(action='RestoreFromClusterSnapshot', verb='POST', path='/', params=params)