import sys
import re
import os
from configparser import RawConfigParser
def parse_variables(config):
    if not config.has_section('variables'):
        raise FormatError('No variables section found !')
    d = {}
    for name, value in config.items('variables'):
        d[name] = value
    return VariableSet(d)