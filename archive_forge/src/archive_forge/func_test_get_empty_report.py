from unittest import mock
from osprofiler.drivers.elasticsearch_driver import ElasticsearchDriver
from osprofiler.tests import test
def test_get_empty_report(self):
    self.elasticsearch.client = mock.MagicMock()
    self.elasticsearch.client.search = mock.MagicMock(return_value={'_scroll_id': '1', 'hits': {'hits': []}})
    self.elasticsearch.client.reset_mock()
    get_report = self.elasticsearch.get_report
    base_id = 'abacaba'
    get_report(base_id)
    self.elasticsearch.client.search.assert_called_once_with(index='osprofiler-notifications', doc_type='notification', size=10000, scroll='2m', body={'query': {'match': {'base_id': base_id}}})