from collections import namedtuple
def get_params_snapshot(self):
    """Return a read-only object of the current cubic parameters.

        These parameters are intended to be used for debug/troubleshooting
        purposes.  These object is a read-only snapshot and cannot be used
        to modify the behavior of the CUBIC calculations.

        New parameters may be added to this object in the future.

        """
    return CubicParams(w_max=self._w_max, k=self._k, last_fail=self._last_fail)