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
def record_testreport(self, testreport: TestReport) -> None:
    names = mangle_test_address(testreport.nodeid)
    existing_attrs = self.attrs
    classnames = names[:-1]
    if self.xml.prefix:
        classnames.insert(0, self.xml.prefix)
    attrs: Dict[str, str] = {'classname': '.'.join(classnames), 'name': bin_xml_escape(names[-1]), 'file': testreport.location[0]}
    if testreport.location[1] is not None:
        attrs['line'] = str(testreport.location[1])
    if hasattr(testreport, 'url'):
        attrs['url'] = testreport.url
    self.attrs = attrs
    self.attrs.update(existing_attrs)
    if self.family == 'xunit1':
        return
    temp_attrs = {}
    for key in self.attrs.keys():
        if key in families[self.family]['testcase']:
            temp_attrs[key] = self.attrs[key]
    self.attrs = temp_attrs