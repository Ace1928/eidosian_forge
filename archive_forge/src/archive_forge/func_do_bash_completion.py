import argparse
import logging
import os
import sys
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
from magnumclient.common import cliutils
from magnumclient import exceptions as exc
from magnumclient.i18n import _
from magnumclient.v1 import client as client_v1
from magnumclient.v1 import shell as shell_v1
from magnumclient import version
def do_bash_completion(self, _args):
    """Prints arguments for bash-completion.

        Prints all of the commands and options to stdout so that the
        magnum.bash_completion script doesn't have to hard code them.
        """
    commands = set()
    options = set()
    for sc_str, sc in self.subcommands.items():
        commands.add(sc_str)
        for option in sc._optionals._option_string_actions.keys():
            options.add(option)
    commands.remove('bash-completion')
    commands.remove('bash_completion')
    print(' '.join(commands | options))