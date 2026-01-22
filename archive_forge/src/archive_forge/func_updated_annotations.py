from pprint import pformat
from six import iteritems
import re
@updated_annotations.setter
def updated_annotations(self, updated_annotations):
    """
        Sets the updated_annotations of this AppsV1beta1DeploymentRollback.
        The annotations to be updated to a deployment

        :param updated_annotations: The updated_annotations of this
        AppsV1beta1DeploymentRollback.
        :type: dict(str, str)
        """
    self._updated_annotations = updated_annotations