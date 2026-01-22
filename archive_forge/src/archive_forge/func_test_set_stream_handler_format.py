import pytest
import logging
from charset_normalizer.utils import set_logging_handler
from charset_normalizer.api import from_bytes, explain_handler
from charset_normalizer.constant import TRACE
def test_set_stream_handler_format(self, caplog):
    set_logging_handler('charset_normalizer', format_string='%(message)s')
    self.logger.info('log content should only be this message')
    assert caplog.record_tuples == [('charset_normalizer', logging.INFO, 'log content should only be this message')]