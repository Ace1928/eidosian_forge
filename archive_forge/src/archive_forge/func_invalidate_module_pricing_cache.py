import re
import os.path
from typing import Dict, Union, Optional
from os.path import join as pjoin
def invalidate_module_pricing_cache(driver_type, driver_name):
    """
    Invalidate the cache for the specified driver.

    :type driver_type: ``str``
    :param driver_type: Driver type ('compute' or 'storage')

    :type driver_name: ``str``
    :param driver_name: Driver name
    """
    if driver_name in PRICING_DATA[driver_type]:
        del PRICING_DATA[driver_type][driver_name]