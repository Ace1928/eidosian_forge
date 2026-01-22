import uuid
import os_traits
from neutron_lib._i18n import _
from neutron_lib import constants as const
from neutron_lib.placement import constants as place_const
def physnet_trait(physnet):
    """A Placement trait name to represent being connected to a physnet.

    :param physnet: The physnet name.
    :returns: The trait name representing the physnet.
    """
    return os_traits.normalize_name('%s%s' % (place_const.TRAIT_PREFIX_PHYSNET, physnet))