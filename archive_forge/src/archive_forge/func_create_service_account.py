import requests
from wandb_gql import gql
from wandb.apis.attrs import Attrs
def create_service_account(self, description):
    """Create a service account for the team.

        Arguments:
            description: (str) A description for this service account

        Returns:
            The service account `Member` object, or None on failure
        """
    try:
        self._client.execute(self.CREATE_SERVICE_ACCOUNT_MUTATION, {'description': description, 'entityName': self.name})
        self.load(True)
        return self.members[-1]
    except requests.exceptions.HTTPError:
        return None