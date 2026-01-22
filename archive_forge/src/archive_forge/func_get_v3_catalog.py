import abc
from keystone.common import provider_api
import keystone.conf
from keystone import exception
def get_v3_catalog(self, user_id, project_id):
    """Retrieve and format the current V3 service catalog.

        Example::

            [
                {
                    "endpoints": [
                    {
                        "interface": "public",
                        "id": "--endpoint-id--",
                        "region": "RegionOne",
                        "url": "http://external:8776/v1/--project-id--"
                    },
                    {
                        "interface": "internal",
                        "id": "--endpoint-id--",
                        "region": "RegionOne",
                        "url": "http://internal:8776/v1/--project-id--"
                    }],
                "id": "--service-id--",
                "type": "volume"
            }]

        :returns: A list representing the service catalog or an empty list
        :raises keystone.exception.NotFound: If the endpoint doesn't exist.

        """
    raise exception.NotImplemented()