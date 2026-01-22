import getpass
import io
import urllib.parse, urllib.request
from warnings import warn
from distutils.core import PyPIRCCommand
from distutils.errors import *
from distutils import log
def build_post_data(self, action):
    meta = self.distribution.metadata
    data = {':action': action, 'metadata_version': '1.0', 'name': meta.get_name(), 'version': meta.get_version(), 'summary': meta.get_description(), 'home_page': meta.get_url(), 'author': meta.get_contact(), 'author_email': meta.get_contact_email(), 'license': meta.get_licence(), 'description': meta.get_long_description(), 'keywords': meta.get_keywords(), 'platform': meta.get_platforms(), 'classifiers': meta.get_classifiers(), 'download_url': meta.get_download_url(), 'provides': meta.get_provides(), 'requires': meta.get_requires(), 'obsoletes': meta.get_obsoletes()}
    if data['provides'] or data['requires'] or data['obsoletes']:
        data['metadata_version'] = '1.1'
    return data