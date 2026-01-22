from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def post_backups_76a17945_3c6f_435c_975b_b5685db10b62_restore(self, **kw):
    return (200, {}, {'restore': _stub_restore()})