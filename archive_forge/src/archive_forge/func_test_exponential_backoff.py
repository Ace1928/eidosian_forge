import mock
from google.auth import _exponential_backoff
@mock.patch('time.sleep', return_value=None)
def test_exponential_backoff(mock_time):
    eb = _exponential_backoff.ExponentialBackoff()
    curr_wait = eb._current_wait_in_seconds
    iteration_count = 0
    for attempt in eb:
        backoff_interval = mock_time.call_args[0][0]
        jitter = curr_wait * eb._randomization_factor
        assert curr_wait - jitter <= backoff_interval <= curr_wait + jitter
        assert attempt == iteration_count + 1
        assert eb.backoff_count == iteration_count + 1
        assert eb._current_wait_in_seconds == eb._multiplier ** (iteration_count + 1)
        curr_wait = eb._current_wait_in_seconds
        iteration_count += 1
    assert eb.total_attempts == _exponential_backoff._DEFAULT_RETRY_TOTAL_ATTEMPTS
    assert eb.backoff_count == _exponential_backoff._DEFAULT_RETRY_TOTAL_ATTEMPTS
    assert iteration_count == _exponential_backoff._DEFAULT_RETRY_TOTAL_ATTEMPTS
    assert mock_time.call_count == _exponential_backoff._DEFAULT_RETRY_TOTAL_ATTEMPTS