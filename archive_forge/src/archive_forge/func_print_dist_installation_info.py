import logging
import shutil
import sys
import textwrap
import xmlrpc.client
from collections import OrderedDict
from optparse import Values
from typing import TYPE_CHECKING, Dict, List, Optional
from pip._vendor.packaging.version import parse as parse_version
from pip._internal.cli.base_command import Command
from pip._internal.cli.req_command import SessionCommandMixin
from pip._internal.cli.status_codes import NO_MATCHES_FOUND, SUCCESS
from pip._internal.exceptions import CommandError
from pip._internal.metadata import get_default_environment
from pip._internal.models.index import PyPI
from pip._internal.network.xmlrpc import PipXmlrpcTransport
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import write_output
def print_dist_installation_info(name: str, latest: str) -> None:
    env = get_default_environment()
    dist = env.get_distribution(name)
    if dist is not None:
        with indent_log():
            if dist.version == latest:
                write_output('INSTALLED: %s (latest)', dist.version)
            else:
                write_output('INSTALLED: %s', dist.version)
                if parse_version(latest).pre:
                    write_output('LATEST:    %s (pre-release; install with `pip install --pre`)', latest)
                else:
                    write_output('LATEST:    %s', latest)