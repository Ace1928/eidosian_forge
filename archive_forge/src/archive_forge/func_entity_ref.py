import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
@property
def entity_ref(self):
    """Entity URI reference."""
    return self._entity_ref