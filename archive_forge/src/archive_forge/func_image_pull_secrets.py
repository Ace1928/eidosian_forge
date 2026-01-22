from pprint import pformat
from six import iteritems
import re
@image_pull_secrets.setter
def image_pull_secrets(self, image_pull_secrets):
    """
        Sets the image_pull_secrets of this V1PodSpec.
        ImagePullSecrets is an optional list of references to secrets in the
        same namespace to use for pulling any of the images used by this
        PodSpec. If specified, these secrets will be passed to individual puller
        implementations for them to use. For example, in the case of docker,
        only DockerConfig type secrets are honored. More info:
        https://kubernetes.io/docs/concepts/containers/images#specifying-imagepullsecrets-on-a-pod

        :param image_pull_secrets: The image_pull_secrets of this V1PodSpec.
        :type: list[V1LocalObjectReference]
        """
    self._image_pull_secrets = image_pull_secrets