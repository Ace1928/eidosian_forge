from ncclient.xml_ import BASE_NS_1_0
from ncclient.operations.errors import OperationError
from .default import DefaultDeviceHandler
def check_device_params(self):
    value = self.device_params.get('with_ns')
    if value in [True, False]:
        return value
    elif value is None:
        return False
    else:
        raise OperationError('Invalid "with_ns" value: %s' % value)