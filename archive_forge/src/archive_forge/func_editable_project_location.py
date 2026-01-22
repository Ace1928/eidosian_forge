import csv
import email.message
import functools
import json
import logging
import pathlib
import re
import zipfile
from typing import (
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.specifiers import InvalidSpecifier, SpecifierSet
from pip._vendor.packaging.utils import NormalizedName, canonicalize_name
from pip._vendor.packaging.version import LegacyVersion, Version
from pip._internal.exceptions import NoneMetadataError
from pip._internal.locations import site_packages, user_site
from pip._internal.models.direct_url import (
from pip._internal.utils.compat import stdlib_pkgs  # TODO: Move definition here.
from pip._internal.utils.egg_link import egg_link_path_from_sys_path
from pip._internal.utils.misc import is_local, normalize_path
from pip._internal.utils.urls import url_to_path
from ._json import msg_to_json
@property
def editable_project_location(self) -> Optional[str]:
    """The project location for editable distributions.

        This is the directory where pyproject.toml or setup.py is located.
        None if the distribution is not installed in editable mode.
        """
    direct_url = self.direct_url
    if direct_url:
        if direct_url.is_local_editable():
            return url_to_path(direct_url.url)
    else:
        egg_link_path = egg_link_path_from_sys_path(self.raw_name)
        if egg_link_path:
            return self.location
    return None