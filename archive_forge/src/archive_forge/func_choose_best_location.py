import copy
from oslo_config import cfg
from oslo_log import log as logging
import stevedore
from glance.i18n import _, _LE
from the first responsive active location it finds in this list.
def choose_best_location(locations, **kwargs):
    """
    Choose best location from image location list by configured strategy.

    :param locations: The original image location list.
    :param kwargs: Strategy-specific arguments for under layer strategy module.
    :returns: The best location from image location list.
    """
    locations = get_ordered_locations(locations, **kwargs)
    if locations:
        return locations[0]
    else:
        return None