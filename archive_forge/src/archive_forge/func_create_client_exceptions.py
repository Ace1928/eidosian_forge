from botocore.exceptions import ClientError
from botocore.utils import get_service_module_name
def create_client_exceptions(self, service_model):
    """Creates a ClientExceptions object for the particular service client

        :type service_model: botocore.model.ServiceModel
        :param service_model: The service model for the client

        :rtype: object that subclasses from BaseClientExceptions
        :returns: The exceptions object of a client that can be used
            to grab the various different modeled exceptions.
        """
    service_name = service_model.service_name
    if service_name not in self._client_exceptions_cache:
        client_exceptions = self._create_client_exceptions(service_model)
        self._client_exceptions_cache[service_name] = client_exceptions
    return self._client_exceptions_cache[service_name]