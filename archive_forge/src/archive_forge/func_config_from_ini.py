import io
import tempfile
import textwrap
import six
from six.moves import configparser
import sys
from pbr.tests import base
from pbr import util
def config_from_ini(ini):
    config = {}
    ini = textwrap.dedent(six.u(ini))
    if sys.version_info >= (3, 2):
        parser = configparser.ConfigParser()
        parser.read_file(io.StringIO(ini))
    else:
        parser = configparser.SafeConfigParser()
        parser.readfp(io.StringIO(ini))
    for section in parser.sections():
        config[section] = dict(parser.items(section))
    return config