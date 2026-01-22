import google.auth
import google.auth.credentials
import google.auth.jwt
import google.auth.transport.grpc
from google.oauth2 import service_account
from google.cloud import pubsub_v1
def test_grpc_request_with_regular_credentials(http_request):
    credentials, project_id = google.auth.default()
    credentials = google.auth.credentials.with_scopes_if_required(credentials, scopes=['https://www.googleapis.com/auth/pubsub'])
    client = pubsub_v1.PublisherClient(credentials=credentials)
    list_topics_iter = client.list_topics(project='projects/{}'.format(project_id))
    list(list_topics_iter)