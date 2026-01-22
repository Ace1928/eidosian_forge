import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
@mock.patch('oslo_vmware.vim_util.continue_retrieval')
@mock.patch('oslo_vmware.vim_util.cancel_retrieval')
def test_with_retrieval(self, cancel_retrieval, continue_retrieval):
    vim = mock.Mock()
    retrieve_result0 = mock.Mock()
    retrieve_result0.objects = [mock.Mock(), mock.Mock()]
    retrieve_result1 = mock.Mock()
    retrieve_result1.objects = [mock.Mock(), mock.Mock()]
    continue_retrieval.side_effect = [retrieve_result1, None]
    expected = retrieve_result0.objects + retrieve_result1.objects
    with vim_util.WithRetrieval(vim, retrieve_result0) as iterator:
        self.assertEqual(expected, list(iterator))
    calls = [mock.call(vim, retrieve_result0), mock.call(vim, retrieve_result1)]
    continue_retrieval.assert_has_calls(calls)
    self.assertFalse(cancel_retrieval.called)