from unittest import mock
from keystoneauth1 import identity
from keystoneauth1 import session
def mock_delete_secret_for_responses(responses, entity_href):
    responses.delete(entity_href, status_code=204)