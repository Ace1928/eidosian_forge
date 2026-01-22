from oslo_serialization import jsonutils
from keystone import exception
from keystone.i18n import _
def translate_urls(json_home, new_prefix):
    """Given a JSON Home document, sticks new_prefix on each of the urls."""
    for dummy_rel, resource in json_home['resources'].items():
        if 'href' in resource:
            resource['href'] = new_prefix + resource['href']
        elif 'href-template' in resource:
            resource['href-template'] = new_prefix + resource['href-template']