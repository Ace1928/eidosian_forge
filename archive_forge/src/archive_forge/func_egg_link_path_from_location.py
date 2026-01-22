import os
import re
import sys
from typing import List, Optional
from pip._internal.locations import site_packages, user_site
from pip._internal.utils.virtualenv import (
def egg_link_path_from_location(raw_name: str) -> Optional[str]:
    """
    Return the path for the .egg-link file if it exists, otherwise, None.

    There's 3 scenarios:
    1) not in a virtualenv
       try to find in site.USER_SITE, then site_packages
    2) in a no-global virtualenv
       try to find in site_packages
    3) in a yes-global virtualenv
       try to find in site_packages, then site.USER_SITE
       (don't look in global location)

    For #1 and #3, there could be odd cases, where there's an egg-link in 2
    locations.

    This method will just return the first one found.
    """
    sites: List[str] = []
    if running_under_virtualenv():
        sites.append(site_packages)
        if not virtualenv_no_global() and user_site:
            sites.append(user_site)
    else:
        if user_site:
            sites.append(user_site)
        sites.append(site_packages)
    egg_link_names = _egg_link_names(raw_name)
    for site in sites:
        for egg_link_name in egg_link_names:
            egglink = os.path.join(site, egg_link_name)
            if os.path.isfile(egglink):
                return egglink
    return None