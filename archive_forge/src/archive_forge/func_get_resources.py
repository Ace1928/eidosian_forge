from lxml import etree
import urllib
from .search import SearchManager
from .users import Users
from .resources import Project
from .tags import Tags
from .jsonutil import JsonTable
def get_resources(self, triple, scan_id):
    return JsonTable(self._intf._get_json('/data/prearchive/projects/%s/scans/%s/resources' % ('/'.join(triple), scan_id))).get('label')