import requests
from wandb_gql import gql
import wandb
from wandb.apis.attrs import Attrs
@property
def user_api(self):
    """An instance of the api using credentials from the user."""
    if self._user_api is None and len(self.api_keys) > 0:
        self._user_api = wandb.Api(api_key=self.api_keys[0])
    return self._user_api