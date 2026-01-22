import logging
from unittest import mock
from osc_lib import logs
from osc_lib.tests import utils
@mock.patch('logging.getLogger')
@mock.patch('osc_lib.logs.set_warning_filter')
def test_init_no_debug(self, warning_filter, getLogger):
    getLogger.side_effect = self.loggers
    self.options.debug = True
    configurator = logs.LogConfigurator(self.options)
    warning_filter.assert_called_with(logging.DEBUG)
    self.requests_log.setLevel.assert_called_with(logging.DEBUG)
    self.assertTrue(configurator.dump_trace)