import optparse
import os
import sys
from subprocess import PIPE, CalledProcessError, Popen
from breezy import osutils
from breezy.tests import ssl_certs
def build_ca_key():
    """Generate an ssl certificate authority private key."""
    key_path = ssl_certs.build_path('ca.key')
    rm_f(key_path)
    _openssl(['genrsa', '-passout', 'stdin', '-des3', '-out', key_path, '4096'], input='%(ca_pass)s\n%(ca_pass)s\n' % ssl_params)