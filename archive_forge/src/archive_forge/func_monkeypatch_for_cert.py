import sys
import os.path
import pkgutil
import shutil
import tempfile
import argparse
import importlib
from base64 import b85decode
def monkeypatch_for_cert(tmpdir):
    """Patches `pip install` to provide default certificate with the lowest priority.

    This ensures that the bundled certificates are used unless the user specifies a
    custom cert via any of pip's option passing mechanisms (config, env-var, CLI).

    A monkeypatch is the easiest way to achieve this, without messing too much with
    the rest of pip's internals.
    """
    from pip._internal.commands.install import InstallCommand
    cert_path = os.path.join(tmpdir, 'cacert.pem')
    with open(cert_path, 'wb') as cert:
        cert.write(pkgutil.get_data('pip._vendor.certifi', 'cacert.pem'))
    install_parse_args = InstallCommand.parse_args

    def cert_parse_args(self, args):
        if not self.parser.get_default_values().cert:
            self.parser.defaults['cert'] = cert_path
        return install_parse_args(self, args)
    InstallCommand.parse_args = cert_parse_args