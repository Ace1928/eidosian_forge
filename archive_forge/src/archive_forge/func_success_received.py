from collections import namedtuple
def success_received(self, timestamp):
    dt = timestamp - self._last_fail
    new_rate = self._scale_constant * (dt - self._k) ** 3 + self._w_max
    return new_rate