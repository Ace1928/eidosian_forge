import ast
from tempest.lib.cli import output_parser
import testtools
from manilaclient import api_versions
from manilaclient import config
def is_microversion_supported(microversion):
    return api_versions.APIVersion(CONF.min_api_microversion) <= api_versions.APIVersion(microversion) <= api_versions.APIVersion(CONF.max_api_microversion)