from unittest import mock
from urllib import parse as urlparse
from heat.api.openstack.v1.views import views_common
from heat.tests import common
def test_get_collection_links_creates_next(self):
    self.setUpGetCollectionLinks()
    links = views_common.get_collection_links(self.request, self.items)
    expected_params = {'marker': ['id2'], 'limit': ['2']}
    next_link = list(filter(lambda link: link['rel'] == 'next', links)).pop()
    self.assertEqual('next', next_link['rel'])
    url_path, url_params = next_link['href'].split('?', 1)
    self.assertEqual(url_path, self.request.path_url)
    self.assertEqual(expected_params, urlparse.parse_qs(url_params))