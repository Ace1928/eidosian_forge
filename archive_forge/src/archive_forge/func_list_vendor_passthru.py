import typing as ty
from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
def list_vendor_passthru(self, session):
    """Fetch vendor specific methods exposed by driver

        :param session: The session to use for making this request.
        :returns: A dict of the available vendor passthru methods for driver.
            Method names keys and corresponding usages in dict form as values
            Usage dict properties:
            * ``async``: bool # Is passthru function invoked asynchronously
            * ``attach``: bool # Is return value attached to response object
            * ``description``: str # Description of what the method does
            * ``http_methods``: list # List of HTTP methods supported
        """
    session = self._get_session(session)
    request = self._prepare_request()
    request.url = utils.urljoin(request.url, 'vendor_passthru', 'methods')
    response = session.get(request.url, headers=request.headers)
    msg = 'Failed to list list vendor_passthru methods for {driver_name}'
    exceptions.raise_from_response(response, error_message=msg.format(driver_name=self.name))
    return response.json()