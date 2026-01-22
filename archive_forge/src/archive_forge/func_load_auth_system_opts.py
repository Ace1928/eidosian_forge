import abc
import argparse
import os
from zunclient.common.apiclient import exceptions
def load_auth_system_opts(parser):
    """Load options needed by the available auth-systems into a parser.

    This function will try to populate the parser with options from the
    available plugins.
    """
    group = parser.add_argument_group('Common auth options')
    BaseAuthPlugin.add_common_opts(group)
    for name, auth_plugin in _discovered_plugins.items():
        group = parser.add_argument_group("Auth-system '%s' options" % name, conflict_handler='resolve')
        auth_plugin.add_opts(group)