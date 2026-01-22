from datetime import datetime
import functools
import os
import platform
import re
from typing import Callable
from typing import Dict
from typing import List
from typing import Match
from typing import Optional
from typing import Tuple
from typing import Union
import xml.etree.ElementTree as ET
from _pytest import nodes
from _pytest import timing
from _pytest._code.code import ExceptionRepr
from _pytest._code.code import ReprFileLocation
from _pytest.config import Config
from _pytest.config import filename_arg
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest
from _pytest.reports import TestReport
from _pytest.stash import StashKey
from _pytest.terminal import TerminalReporter
import pytest
@pytest.fixture
def record_xml_attribute(request: FixtureRequest) -> Callable[[str, object], None]:
    """Add extra xml attributes to the tag for the calling test.

    The fixture is callable with ``name, value``. The value is
    automatically XML-encoded.
    """
    from _pytest.warning_types import PytestExperimentalApiWarning
    request.node.warn(PytestExperimentalApiWarning('record_xml_attribute is an experimental feature'))
    _warn_incompatibility_with_xunit2(request, 'record_xml_attribute')

    def add_attr_noop(name: str, value: object) -> None:
        pass
    attr_func = add_attr_noop
    xml = request.config.stash.get(xml_key, None)
    if xml is not None:
        node_reporter = xml.node_reporter(request.node.nodeid)
        attr_func = node_reporter.add_attribute
    return attr_func