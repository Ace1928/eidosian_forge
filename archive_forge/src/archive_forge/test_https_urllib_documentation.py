import os
import sys
from .. import config, tests, trace
from ..transport.http import opt_ssl_ca_certs, ssl
Check that the default we provide exists for the tested platform.