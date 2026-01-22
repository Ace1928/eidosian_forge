import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.redshift import exceptions
def create_cluster_snapshot(self, snapshot_identifier, cluster_identifier):
    """
        Creates a manual snapshot of the specified cluster. The
        cluster must be in the `available` state.

        For more information about working with snapshots, go to
        `Amazon Redshift Snapshots`_ in the Amazon Redshift Management
        Guide .

        :type snapshot_identifier: string
        :param snapshot_identifier: A unique identifier for the snapshot that
            you are requesting. This identifier must be unique for all
            snapshots within the AWS account.
        Constraints:


        + Cannot be null, empty, or blank
        + Must contain from 1 to 255 alphanumeric characters or hyphens
        + First character must be a letter
        + Cannot end with a hyphen or contain two consecutive hyphens


        Example: `my-snapshot-id`

        :type cluster_identifier: string
        :param cluster_identifier: The cluster identifier for which you want a
            snapshot.

        """
    params = {'SnapshotIdentifier': snapshot_identifier, 'ClusterIdentifier': cluster_identifier}
    return self._make_request(action='CreateClusterSnapshot', verb='POST', path='/', params=params)