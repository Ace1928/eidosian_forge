import google
import google.oauth2.credentials
from google.auth import compute_engine
import google.auth.transport.requests

    Use the Google Cloud metadata server in the Cloud Run (or AppEngine or Kubernetes etc.,)
    environment to create an identity token and add it to the HTTP request as part of an
    Authorization header.

    Args:
        url: The url or target audience to obtain the ID token for.
            Examples: http://www.abc.com
    