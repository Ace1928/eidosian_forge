from pprint import pformat
from six import iteritems
import re
@volume_claim_templates.setter
def volume_claim_templates(self, volume_claim_templates):
    """
        Sets the volume_claim_templates of this V1beta2StatefulSetSpec.
        volumeClaimTemplates is a list of claims that pods are allowed to
        reference. The StatefulSet controller is responsible for mapping network
        identities to claims in a way that maintains the identity of a pod.
        Every claim in this list must have at least one matching (by name)
        volumeMount in one container in the template. A claim in this list takes
        precedence over any volumes in the template, with the same name.

        :param volume_claim_templates: The volume_claim_templates of this
        V1beta2StatefulSetSpec.
        :type: list[V1PersistentVolumeClaim]
        """
    self._volume_claim_templates = volume_claim_templates