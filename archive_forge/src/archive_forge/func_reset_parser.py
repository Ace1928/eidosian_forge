from __future__ import absolute_import
from ruamel.yaml.error import MarkedYAMLError
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.scanner import Scanner, RoundTripScanner, ScannerError  # NOQA
from ruamel.yaml.compat import utf8, nprint, nprintf  # NOQA
def reset_parser(self):
    self.current_event = None
    self.yaml_version = None
    self.tag_handles = {}
    self.states = []
    self.marks = []
    self.state = self.parse_stream_start