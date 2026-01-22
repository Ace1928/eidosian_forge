from heat.api.aws import exception as aws_exception
from heat.api.aws import utils as api_utils
from heat.common import exception as common_exception
from heat.tests import common
def test_map_remote_error_request_limit_exceeded(self):
    ex = common_exception.RequestLimitExceeded(message='testing')
    expected = aws_exception.HeatRequestLimitExceeded
    self.assertIsInstance(aws_exception.map_remote_error(ex), expected)