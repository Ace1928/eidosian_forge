from __future__ import absolute_import, unicode_literals, print_function
import sys
import os
import warnings
import glob
from importlib import import_module
import ruamel.yaml
from ruamel.yaml.error import UnsafeLoaderWarning, YAMLError  # NOQA
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.nodes import *  # NOQA
from ruamel.yaml.loader import BaseLoader, SafeLoader, Loader, RoundTripLoader  # NOQA
from ruamel.yaml.dumper import BaseDumper, SafeDumper, Dumper, RoundTripDumper  # NOQA
from ruamel.yaml.compat import StringIO, BytesIO, with_metaclass, PY3, nprint
from ruamel.yaml.resolver import VersionedResolver, Resolver  # NOQA
from ruamel.yaml.representer import (
from ruamel.yaml.constructor import (
from ruamel.yaml.loader import Loader as UnsafeLoader
def get_constructor_parser(self, stream):
    """
        the old cyaml needs special setup, and therefore the stream
        """
    if self.Parser is not CParser:
        if self.Reader is None:
            self.Reader = ruamel.yaml.reader.Reader
        if self.Scanner is None:
            self.Scanner = ruamel.yaml.scanner.Scanner
        self.reader.stream = stream
    elif self.Reader is not None:
        if self.Scanner is None:
            self.Scanner = ruamel.yaml.scanner.Scanner
        self.Parser = ruamel.yaml.parser.Parser
        self.reader.stream = stream
    elif self.Scanner is not None:
        if self.Reader is None:
            self.Reader = ruamel.yaml.reader.Reader
        self.Parser = ruamel.yaml.parser.Parser
        self.reader.stream = stream
    else:
        rslvr = self.Resolver

        class XLoader(self.Parser, self.Constructor, rslvr):

            def __init__(selfx, stream, version=self.version, preserve_quotes=None):
                CParser.__init__(selfx, stream)
                selfx._parser = selfx._composer = selfx
                self.Constructor.__init__(selfx, loader=selfx)
                selfx.allow_duplicate_keys = self.allow_duplicate_keys
                rslvr.__init__(selfx, version=version, loadumper=selfx)
        self._stream = stream
        loader = XLoader(stream)
        return (loader, loader)
    return (self.constructor, self.parser)