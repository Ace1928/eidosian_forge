import json
import logging
import os
import shlex
import subprocess
from tempest.lib.cli import output_parser
from tempest.lib import exceptions
import testtools
@classmethod
def get_openstack_configuration_value(cls, configuration):
    opts = cls.get_opts([configuration])
    return cls.openstack('configuration show ' + opts)