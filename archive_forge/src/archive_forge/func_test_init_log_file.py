import logging
from unittest import mock
from osc_lib import logs
from osc_lib.tests import utils
@mock.patch('logging.FileHandler')
@mock.patch('logging.getLogger')
@mock.patch('osc_lib.logs.set_warning_filter')
@mock.patch('osc_lib.logs._FileFormatter')
def test_init_log_file(self, formatter, warning_filter, getLogger, handle):
    getLogger.side_effect = self.loggers
    self.options.log_file = '/tmp/log_file'
    file_logger = mock.Mock()
    file_logger.setFormatter = mock.Mock()
    file_logger.setLevel = mock.Mock()
    handle.return_value = file_logger
    mock_formatter = mock.Mock()
    formatter.return_value = mock_formatter
    logs.LogConfigurator(self.options)
    handle.assert_called_with(filename=self.options.log_file)
    self.root_logger.addHandler.assert_called_with(file_logger)
    file_logger.setFormatter.assert_called_with(mock_formatter)
    file_logger.setLevel.assert_called_with(logging.WARNING)