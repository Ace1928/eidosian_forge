import configparser
import copy
import os
import shlex
import sys
import botocore.exceptions
def build_profile_map(parsed_ini_config):
    """Convert the parsed INI config into a profile map.

    The config file format requires that every profile except the
    default to be prepended with "profile", e.g.::

        [profile test]
        aws_... = foo
        aws_... = bar

        [profile bar]
        aws_... = foo
        aws_... = bar

        # This is *not* a profile
        [preview]
        otherstuff = 1

        # Neither is this
        [foobar]
        morestuff = 2

    The build_profile_map will take a parsed INI config file where each top
    level key represents a section name, and convert into a format where all
    the profiles are under a single top level "profiles" key, and each key in
    the sub dictionary is a profile name.  For example, the above config file
    would be converted from::

        {"profile test": {"aws_...": "foo", "aws...": "bar"},
         "profile bar": {"aws...": "foo", "aws...": "bar"},
         "preview": {"otherstuff": ...},
         "foobar": {"morestuff": ...},
         }

    into::

        {"profiles": {"test": {"aws_...": "foo", "aws...": "bar"},
                      "bar": {"aws...": "foo", "aws...": "bar"},
         "preview": {"otherstuff": ...},
         "foobar": {"morestuff": ...},
        }

    If there are no profiles in the provided parsed INI contents, then
    an empty dict will be the value associated with the ``profiles`` key.

    .. note::

        This will not mutate the passed in parsed_ini_config.  Instead it will
        make a deepcopy and return that value.

    """
    parsed_config = copy.deepcopy(parsed_ini_config)
    profiles = {}
    sso_sessions = {}
    services = {}
    final_config = {}
    for key, values in parsed_config.items():
        if key.startswith('profile'):
            profiles.update(_parse_section(key, values))
        elif key.startswith('sso-session'):
            sso_sessions.update(_parse_section(key, values))
        elif key.startswith('services'):
            services.update(_parse_section(key, values))
        elif key == 'default':
            profiles[key] = values
        else:
            final_config[key] = values
    final_config['profiles'] = profiles
    final_config['sso_sessions'] = sso_sessions
    final_config['services'] = services
    return final_config