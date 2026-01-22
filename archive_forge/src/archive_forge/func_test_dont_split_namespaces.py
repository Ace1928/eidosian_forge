import textwrap
from unittest import mock
from oslotest import base
from oslo_config import cfg
from oslo_config import sphinxext
@mock.patch('oslo_config.generator._list_opts')
@mock.patch('oslo_config.sphinxext._format_group_opts')
def test_dont_split_namespaces(self, _format_group_opts, _list_opts):
    _list_opts.return_value = [('namespace1', [(None, ['opt1'])]), ('namespace2', [(None, ['opt2'])])]
    list(sphinxext._format_option_help(namespaces=['namespace1', 'namespace2'], split_namespaces=False))
    _format_group_opts.assert_called_once_with(namespace=None, group_name='DEFAULT', group_obj=None, opt_list=['opt1', 'opt2'])