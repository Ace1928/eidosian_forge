from ncclient.operations.third_party.huawei.rpc import *
from ncclient.xml_ import BASE_NS_1_0
from .default import DefaultDeviceHandler
def handle_raw_dispatch(self, raw):
    return raw.strip('\x00')