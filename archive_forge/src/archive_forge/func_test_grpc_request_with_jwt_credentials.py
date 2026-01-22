import google.auth
import google.auth.credentials
import google.auth.jwt
import google.auth.transport.grpc
from google.oauth2 import service_account
from google.cloud import pubsub_v1
def test_grpc_request_with_jwt_credentials():
    credentials, project_id = google.auth.default()
    audience = 'https://pubsub.googleapis.com/google.pubsub.v1.Publisher'
    credentials = google.auth.jwt.Credentials.from_signing_credentials(credentials, audience=audience)
    client = pubsub_v1.PublisherClient(credentials=credentials)
    list_topics_iter = client.list_topics(project='projects/{}'.format(project_id))
    list(list_topics_iter)