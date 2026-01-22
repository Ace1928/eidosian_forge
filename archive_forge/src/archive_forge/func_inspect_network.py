from ..errors import InvalidVersion
from ..utils import check_resource, minimum_version
from ..utils import version_lt
from .. import utils
@check_resource('net_id')
def inspect_network(self, net_id, verbose=None, scope=None):
    """
        Get detailed information about a network.

        Args:
            net_id (str): ID of network
            verbose (bool): Show the service details across the cluster in
                swarm mode.
            scope (str): Filter the network by scope (``swarm``, ``global``
                or ``local``).
        """
    params = {}
    if verbose is not None:
        if version_lt(self._version, '1.28'):
            raise InvalidVersion('verbose was introduced in API 1.28')
        params['verbose'] = verbose
    if scope is not None:
        if version_lt(self._version, '1.31'):
            raise InvalidVersion('scope was introduced in API 1.31')
        params['scope'] = scope
    url = self._url('/networks/{0}', net_id)
    res = self._get(url, params=params)
    return self._result(res, json=True)