import copy
from oslo_config import cfg
from oslo_log import log as logging
import stevedore
from glance.i18n import _, _LE
from the first responsive active location it finds in this list.
def verify_location_strategy(conf=None, strategies=_available_strategies):
    """Validate user configured 'location_strategy' option value."""
    if not conf:
        conf = CONF.location_strategy
    if conf not in strategies:
        msg = _('Invalid location_strategy option: %(name)s. The valid strategy option(s) is(are): %(strategies)s') % {'name': conf, 'strategies': ', '.join(strategies.keys())}
        LOG.error(msg)
        raise RuntimeError(msg)