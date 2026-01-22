from pprint import pformat
from six import iteritems
import re
@service_account_name.setter
def service_account_name(self, service_account_name):
    """
        Sets the service_account_name of this V1PodSpec.
        ServiceAccountName is the name of the ServiceAccount to use to run this
        pod. More info:
        https://kubernetes.io/docs/tasks/configure-pod-container/configure-service-account/

        :param service_account_name: The service_account_name of this V1PodSpec.
        :type: str
        """
    self._service_account_name = service_account_name