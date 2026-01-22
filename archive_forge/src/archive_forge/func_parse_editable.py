import copy
import logging
import os
import re
from typing import Collection, Dict, List, Optional, Set, Tuple, Union
from pip._vendor.packaging.markers import Marker
from pip._vendor.packaging.requirements import InvalidRequirement, Requirement
from pip._vendor.packaging.specifiers import Specifier
from pip._internal.exceptions import InstallationError
from pip._internal.models.index import PyPI, TestPyPI
from pip._internal.models.link import Link
from pip._internal.models.wheel import Wheel
from pip._internal.req.req_file import ParsedRequirement
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.filetypes import is_archive_file
from pip._internal.utils.misc import is_installable_dir
from pip._internal.utils.packaging import get_requirement
from pip._internal.utils.urls import path_to_url
from pip._internal.vcs import is_url, vcs
def parse_editable(editable_req: str) -> Tuple[Optional[str], str, Set[str]]:
    """Parses an editable requirement into:
        - a requirement name
        - an URL
        - extras
        - editable options
    Accepted requirements:
        svn+http://blahblah@rev#egg=Foobar[baz]&subdirectory=version_subdir
        .[some_extra]
    """
    url = editable_req
    url_no_extras, extras = _strip_extras(url)
    if os.path.isdir(url_no_extras):
        url_no_extras = path_to_url(url_no_extras)
    if url_no_extras.lower().startswith('file:'):
        package_name = Link(url_no_extras).egg_fragment
        if extras:
            return (package_name, url_no_extras, get_requirement('placeholder' + extras.lower()).extras)
        else:
            return (package_name, url_no_extras, set())
    for version_control in vcs:
        if url.lower().startswith(f'{version_control}:'):
            url = f'{version_control}+{url}'
            break
    link = Link(url)
    if not link.is_vcs:
        backends = ', '.join(vcs.all_schemes)
        raise InstallationError(f'{editable_req} is not a valid editable requirement. It should either be a path to a local project or a VCS URL (beginning with {backends}).')
    package_name = link.egg_fragment
    if not package_name:
        raise InstallationError("Could not detect requirement name for '{}', please specify one with #egg=your_package_name".format(editable_req))
    return (package_name, url, set())