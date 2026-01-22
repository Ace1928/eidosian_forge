from pprint import pformat
from six import iteritems
import re
@readiness_probe.setter
def readiness_probe(self, readiness_probe):
    """
        Sets the readiness_probe of this V1Container.
        Periodic probe of container service readiness. Container will be removed
        from service endpoints if the probe fails. Cannot be updated. More info:
        https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle#container-probes

        :param readiness_probe: The readiness_probe of this V1Container.
        :type: V1Probe
        """
    self._readiness_probe = readiness_probe