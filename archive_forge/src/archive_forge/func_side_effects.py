from pprint import pformat
from six import iteritems
import re
@side_effects.setter
def side_effects(self, side_effects):
    """
        Sets the side_effects of this V1beta1Webhook.
        SideEffects states whether this webhookk has side effects. Acceptable
        values are: Unknown, None, Some, NoneOnDryRun Webhooks with side effects
        MUST implement a reconciliation system, since a request may be rejected
        by a future step in the admission change and the side effects therefore
        need to be undone. Requests with the dryRun attribute will be
        auto-rejected if they match a webhook with sideEffects == Unknown or
        Some. Defaults to Unknown.

        :param side_effects: The side_effects of this V1beta1Webhook.
        :type: str
        """
    self._side_effects = side_effects