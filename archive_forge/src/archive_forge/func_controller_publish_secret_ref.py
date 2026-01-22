from pprint import pformat
from six import iteritems
import re
@controller_publish_secret_ref.setter
def controller_publish_secret_ref(self, controller_publish_secret_ref):
    """
        Sets the controller_publish_secret_ref of this
        V1CSIPersistentVolumeSource.
        ControllerPublishSecretRef is a reference to the secret object
        containing sensitive information to pass to the CSI driver to complete
        the CSI ControllerPublishVolume and ControllerUnpublishVolume calls.
        This field is optional, and may be empty if no secret is required. If
        the secret object contains more than one secret, all secrets are passed.

        :param controller_publish_secret_ref: The controller_publish_secret_ref
        of this V1CSIPersistentVolumeSource.
        :type: V1SecretReference
        """
    self._controller_publish_secret_ref = controller_publish_secret_ref