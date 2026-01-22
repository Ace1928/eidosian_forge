import itertools
from heat.common import exception
from heat.engine import attributes
from heat.engine import status
def get_availability_zones(self):
    """Return the list of Nova availability zones."""
    if self._zones is None:
        nova = self._context.clients.client('nova')
        zones = nova.availability_zones.list(detailed=False)
        self._zones = [zone.zoneName for zone in zones]
    return self._zones