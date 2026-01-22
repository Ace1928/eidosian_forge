import http.client as http
import ddt
import webob
from oslo_serialization import jsonutils
from glance.api.middleware import version_negotiation
from glance.api import versions
from glance.tests.unit import base
def get_versions_list(url, enabled_backends=False, enabled_cache=False):
    image_versions = [{'id': 'v2.15', 'status': 'CURRENT', 'links': [{'rel': 'self', 'href': '%s/v2/' % url}]}, {'id': 'v2.9', 'status': 'SUPPORTED', 'links': [{'rel': 'self', 'href': '%s/v2/' % url}]}, {'id': 'v2.7', 'status': 'SUPPORTED', 'links': [{'rel': 'self', 'href': '%s/v2/' % url}]}, {'id': 'v2.6', 'status': 'SUPPORTED', 'links': [{'rel': 'self', 'href': '%s/v2/' % url}]}, {'id': 'v2.5', 'status': 'SUPPORTED', 'links': [{'rel': 'self', 'href': '%s/v2/' % url}]}, {'id': 'v2.4', 'status': 'SUPPORTED', 'links': [{'rel': 'self', 'href': '%s/v2/' % url}]}, {'id': 'v2.3', 'status': 'SUPPORTED', 'links': [{'rel': 'self', 'href': '%s/v2/' % url}]}, {'id': 'v2.2', 'status': 'SUPPORTED', 'links': [{'rel': 'self', 'href': '%s/v2/' % url}]}, {'id': 'v2.1', 'status': 'SUPPORTED', 'links': [{'rel': 'self', 'href': '%s/v2/' % url}]}, {'id': 'v2.0', 'status': 'SUPPORTED', 'links': [{'rel': 'self', 'href': '%s/v2/' % url}]}]
    if enabled_backends:
        image_versions = [{'id': 'v2.15', 'status': 'CURRENT', 'links': [{'rel': 'self', 'href': '%s/v2/' % url}]}, {'id': 'v2.13', 'status': 'SUPPORTED', 'links': [{'rel': 'self', 'href': '%s/v2/' % url}]}, {'id': 'v2.12', 'status': 'SUPPORTED', 'links': [{'rel': 'self', 'href': '%s/v2/' % url}]}, {'id': 'v2.11', 'status': 'SUPPORTED', 'links': [{'rel': 'self', 'href': '%s/v2/' % url}]}, {'id': 'v2.10', 'status': 'SUPPORTED', 'links': [{'rel': 'self', 'href': '%s/v2/' % url}]}, {'id': 'v2.9', 'status': 'SUPPORTED', 'links': [{'rel': 'self', 'href': '%s/v2/' % url}]}, {'id': 'v2.8', 'status': 'SUPPORTED', 'links': [{'rel': 'self', 'href': '%s/v2/' % url}]}] + image_versions[2:]
    if enabled_cache:
        image_versions[0]['status'] = 'SUPPORTED'
        image_versions.insert(1, {'id': 'v2.14', 'status': 'SUPPORTED', 'links': [{'rel': 'self', 'href': '%s/v2/' % url}]})
        image_versions.insert(0, {'id': 'v2.16', 'status': 'CURRENT', 'links': [{'rel': 'self', 'href': '%s/v2/' % url}]})
    return image_versions