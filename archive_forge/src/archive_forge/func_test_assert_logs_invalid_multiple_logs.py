import logging
import warnings
import pytest
import cirq.testing
def test_assert_logs_invalid_multiple_logs():
    with pytest.raises(AssertionError, match='^Expected 1 log message but got 2. Log messages.*$'):
        with cirq.testing.assert_logs('dog'):
            logging.error('orange apple fruit')
            logging.error('dog')
    with pytest.raises(AssertionError, match='^Expected 2 log message but got 3. Log messages.*$'):
        with cirq.testing.assert_logs('dog', count=2):
            logging.error('orange apple fruit')
            logging.error('other')
            logging.error('dog')
    match = "^dog expected to appear in log messages but it was not found. Log messages: \\['orange', 'other', 'whatever'\\].$"
    with pytest.raises(AssertionError, match=match):
        with cirq.testing.assert_logs('dog', count=3):
            logging.error('orange')
            logging.error('other')
            logging.error('whatever')