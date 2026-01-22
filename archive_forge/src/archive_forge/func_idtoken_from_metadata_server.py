import google
import google.oauth2.credentials
from google.auth import compute_engine
import google.auth.transport.requests
def idtoken_from_metadata_server(url: str):
    """
    Use the Google Cloud metadata server in the Cloud Run (or AppEngine or Kubernetes etc.,)
    environment to create an identity token and add it to the HTTP request as part of an
    Authorization header.

    Args:
        url: The url or target audience to obtain the ID token for.
            Examples: http://www.abc.com
    """
    request = google.auth.transport.requests.Request()
    credentials = compute_engine.IDTokenCredentials(request=request, target_audience=url, use_metadata_identity_endpoint=True)
    credentials.refresh(request)
    print('Generated ID token.')