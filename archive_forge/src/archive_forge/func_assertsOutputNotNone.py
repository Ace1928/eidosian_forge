import json
import logging
import os
import shlex
import subprocess
from tempest.lib.cli import output_parser
from tempest.lib import exceptions
import testtools
@classmethod
def assertsOutputNotNone(cls, observed):
    if observed is None:
        raise Exception('No output observed')