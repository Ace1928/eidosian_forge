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
def install_req_drop_extras(ireq: InstallRequirement) -> InstallRequirement:
    """
    Creates a new InstallationRequirement using the given template but without
    any extras. Sets the original requirement as the new one's parent
    (comes_from).
    """
    return InstallRequirement(req=_set_requirement_extras(ireq.req, set()) if ireq.req is not None else None, comes_from=ireq, editable=ireq.editable, link=ireq.link, markers=ireq.markers, use_pep517=ireq.use_pep517, isolated=ireq.isolated, global_options=ireq.global_options, hash_options=ireq.hash_options, constraint=ireq.constraint, extras=[], config_settings=ireq.config_settings, user_supplied=ireq.user_supplied, permit_editable_wheels=ireq.permit_editable_wheels)