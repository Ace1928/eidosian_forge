from pprint import pformat
from six import iteritems
import re
@rollback_to.setter
def rollback_to(self, rollback_to):
    """
        Sets the rollback_to of this AppsV1beta1DeploymentRollback.
        The config of this deployment rollback.

        :param rollback_to: The rollback_to of this
        AppsV1beta1DeploymentRollback.
        :type: AppsV1beta1RollbackConfig
        """
    if rollback_to is None:
        raise ValueError('Invalid value for `rollback_to`, must not be `None`')
    self._rollback_to = rollback_to