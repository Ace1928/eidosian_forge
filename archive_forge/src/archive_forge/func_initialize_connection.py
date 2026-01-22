from cinderclient.apiclient import base as common_base
from cinderclient import base
def initialize_connection(self, volume, connector):
    """Initialize a volume connection.

        :param volume: The :class:`Volume` (or its ID).
        :param connector: connector dict from nova.
        """
    resp, body = self._action('os-initialize_connection', volume, {'connector': connector})
    return common_base.DictWithMeta(body['connection_info'], resp)