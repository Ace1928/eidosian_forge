from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
@api_versions.wraps(constants.REPLICA_GRADUATION_VERSION)
def resync(self, replica):
    """Re-sync the provided replica.

        :param replica: either replica object or its UUID.
        """
    return self._action('resync', replica)