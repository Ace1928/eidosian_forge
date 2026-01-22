from pprint import pformat
from six import iteritems
import re
@image_pull_policy.setter
def image_pull_policy(self, image_pull_policy):
    """
        Sets the image_pull_policy of this V1Container.
        Image pull policy. One of Always, Never, IfNotPresent. Defaults to
        Always if :latest tag is specified, or IfNotPresent otherwise. Cannot be
        updated. More info:
        https://kubernetes.io/docs/concepts/containers/images#updating-images

        :param image_pull_policy: The image_pull_policy of this V1Container.
        :type: str
        """
    self._image_pull_policy = image_pull_policy