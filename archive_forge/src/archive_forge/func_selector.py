from pprint import pformat
from six import iteritems
import re
@selector.setter
def selector(self, selector):
    """
        Sets the selector of this V1beta2StatefulSetSpec.
        selector is a label query over pods that should match the replica count.
        It must match the pod template's labels. More info:
        https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/#label-selectors

        :param selector: The selector of this V1beta2StatefulSetSpec.
        :type: V1LabelSelector
        """
    if selector is None:
        raise ValueError('Invalid value for `selector`, must not be `None`')
    self._selector = selector