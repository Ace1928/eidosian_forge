import uuid
import os_traits
from neutron_lib._i18n import _
from neutron_lib import constants as const
from neutron_lib.placement import constants as place_const
def parse_rp_inventory_defaults(inventory_defaults):
    """Parse and validate config option: parse_rp_inventory_defaults.

    Cast the dict values to the proper numerical types.

    Input in the config:
        resource_provider_inventory_defaults = allocation_ratio:1.0,min_unit:1
    Input here:
        {
            'allocation_ratio': '1.0',
            'min_unit': '1',
        }
    Output here:
        {
            'allocation_ratio': 1.0,
            'min_unit': 1,
        }

    :param inventory_defaults: The dict of inventory parameters and values (as
                               strings) as pre-parsed by oslo_config.
    :raises: ValueError on invalid input.
    :returns: The fully parsed inventory parameters and values (as numerical
              values) as a dict.
    """
    unexpected_options = set(inventory_defaults.keys()) - place_const.INVENTORY_OPTIONS
    if unexpected_options:
        raise ValueError(_('Cannot parse inventory_defaults. Unexpected options: %s') % ','.join(unexpected_options))
    try:
        if 'allocation_ratio' in inventory_defaults:
            inventory_defaults['allocation_ratio'] = float(inventory_defaults['allocation_ratio'])
            if inventory_defaults['allocation_ratio'] < 0:
                raise ValueError()
    except ValueError as e:
        raise ValueError(_('Cannot parse inventory_defaults.allocation_ratio. Expected: non-negative float, got: %s') % inventory_defaults['allocation_ratio']) from e
    for key in ('min_unit', 'max_unit', 'reserved', 'step_size'):
        try:
            if key in inventory_defaults:
                inventory_defaults[key] = int(inventory_defaults[key])
                if inventory_defaults[key] < 0:
                    raise ValueError()
        except ValueError as e:
            raise ValueError(_('Cannot parse inventory_defaults.%(key)s. Expected: non-negative int, got: %(got)s') % {'key': key, 'got': inventory_defaults[key]}) from e
    return inventory_defaults