from pprint import pformat
from six import iteritems
import re
@webhooks.setter
def webhooks(self, webhooks):
    """
        Sets the webhooks of this V1beta1ValidatingWebhookConfiguration.
        Webhooks is a list of webhooks and the affected resources and
        operations.

        :param webhooks: The webhooks of this
        V1beta1ValidatingWebhookConfiguration.
        :type: list[V1beta1Webhook]
        """
    self._webhooks = webhooks