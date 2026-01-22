import requests
from wandb_gql import gql
import wandb
from wandb.apis.attrs import Attrs
def delete_api_key(self, api_key):
    """Delete a user's api key.

        Returns:
            Boolean indicating success

        Raises:
            ValueError if the api_key couldn't be found
        """
    idx = self.api_keys.index(api_key)
    try:
        self._client.execute(self.DELETE_API_KEY_MUTATION, {'id': self._attrs['apiKeys']['edges'][idx]['node']['id']})
    except requests.exceptions.HTTPError:
        return False
    return True