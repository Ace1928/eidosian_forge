import re
import argparse
import os
import fileinput
import logging
from xmlschema.cli import xsd_version_number, defuse_data
from xmlschema.validators import XMLSchema10, XMLSchema11
from ._observers import ObservedXMLSchema10, ObservedXMLSchema11
def get_test_args(args_line):
    """Returns the list of arguments from provided text line."""
    try:
        args_line, _ = args_line.split('#', 1)
    except ValueError:
        pass
    return re.split('(?<!\\\\) ', args_line.strip())