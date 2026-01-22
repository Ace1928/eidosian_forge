import pkg_resources
import argparse
import logging
import sys
from warnings import warn
@classmethod
def handle_command_line(cls):
    runner = CommandRunner()
    runner.run(sys.argv[1:])