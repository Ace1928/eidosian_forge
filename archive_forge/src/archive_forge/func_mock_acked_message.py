from taskflow.engines.worker_based import dispatcher
from taskflow import test
from taskflow.test import mock
def mock_acked_message(ack_ok=True, **kwargs):
    msg = mock.create_autospec(message.Message, spec_set=True, instance=True, channel=None, **kwargs)

    def ack_side_effect(*args, **kwargs):
        msg.acknowledged = True
    if ack_ok:
        msg.ack_log_error.side_effect = ack_side_effect
    msg.acknowledged = False
    return msg