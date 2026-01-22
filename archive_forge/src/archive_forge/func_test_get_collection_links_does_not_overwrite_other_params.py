from unittest import mock
from urllib import parse as urlparse
from heat.api.openstack.v1.views import views_common
from heat.tests import common
def test_get_collection_links_does_not_overwrite_other_params(self):
    self.setUpGetCollectionLinks()
    self.request.params = {'limit': '2', 'foo': 'bar'}
    links = views_common.get_collection_links(self.request, self.items)
    next_link = list(filter(lambda link: link['rel'] == 'next', links)).pop()
    url = next_link['href']
    query_string = urlparse.urlparse(url).query
    params = {}
    params.update(urlparse.parse_qsl(query_string))
    self.assertEqual('2', params['limit'])
    self.assertEqual('bar', params['foo'])