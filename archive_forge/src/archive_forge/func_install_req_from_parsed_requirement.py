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
def install_req_from_parsed_requirement(parsed_req: ParsedRequirement, isolated: bool=False, use_pep517: Optional[bool]=None, user_supplied: bool=False, config_settings: Optional[Dict[str, Union[str, List[str]]]]=None) -> InstallRequirement:
    if parsed_req.is_editable:
        req = install_req_from_editable(parsed_req.requirement, comes_from=parsed_req.comes_from, use_pep517=use_pep517, constraint=parsed_req.constraint, isolated=isolated, user_supplied=user_supplied, config_settings=config_settings)
    else:
        req = install_req_from_line(parsed_req.requirement, comes_from=parsed_req.comes_from, use_pep517=use_pep517, isolated=isolated, global_options=parsed_req.options.get('global_options', []) if parsed_req.options else [], hash_options=parsed_req.options.get('hashes', {}) if parsed_req.options else {}, constraint=parsed_req.constraint, line_source=parsed_req.line_source, user_supplied=user_supplied, config_settings=config_settings)
    return req