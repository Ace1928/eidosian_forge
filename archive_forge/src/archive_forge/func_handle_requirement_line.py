import logging
import optparse
import os
import re
import shlex
import urllib.parse
from optparse import Values
from typing import (
from pip._internal.cli import cmdoptions
from pip._internal.exceptions import InstallationError, RequirementsFileParseError
from pip._internal.models.search_scope import SearchScope
from pip._internal.network.session import PipSession
from pip._internal.network.utils import raise_for_status
from pip._internal.utils.encoding import auto_decode
from pip._internal.utils.urls import get_url_scheme
def handle_requirement_line(line: ParsedLine, options: Optional[optparse.Values]=None) -> ParsedRequirement:
    line_comes_from = '{} {} (line {})'.format('-c' if line.constraint else '-r', line.filename, line.lineno)
    assert line.is_requirement
    if line.is_editable:
        supported_dest = SUPPORTED_OPTIONS_EDITABLE_REQ_DEST
    else:
        supported_dest = SUPPORTED_OPTIONS_REQ_DEST
    req_options = {}
    for dest in supported_dest:
        if dest in line.opts.__dict__ and line.opts.__dict__[dest]:
            req_options[dest] = line.opts.__dict__[dest]
    line_source = f'line {line.lineno} of {line.filename}'
    return ParsedRequirement(requirement=line.requirement, is_editable=line.is_editable, comes_from=line_comes_from, constraint=line.constraint, options=req_options, line_source=line_source)