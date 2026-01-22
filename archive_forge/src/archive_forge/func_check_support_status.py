from ... import version_info  # noqa: F401
from ... import controldir, errors
from ... import transport as _mod_transport
def check_support_status(self, allow_unsupported, recommend_upgrade=True, basedir=None):
    raise MercurialUnsupportedError()