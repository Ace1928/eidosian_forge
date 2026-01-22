import enum
import threading
from cupyx.distributed import _klv_utils
def decode_result(self, data):
    return Barrier.BarrierResult.from_klv(data)