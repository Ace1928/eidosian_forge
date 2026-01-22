from cinderclient import api_versions
from cinderclient.apiclient import base as common_base
from cinderclient import base
from cinderclient.v3 import volumes_base
@api_versions.wraps('3.71')
def extend_volume_completion(self, volume, error=False):
    """Complete extending an attached volume.

        :param volume: The UUID of the extended volume
        :param error: Used to indicate if an error has occured that requires
                      Cinder to roll back the extend operation.
        """
    return self._action('os-extend_volume_completion', volume, {'error': error})