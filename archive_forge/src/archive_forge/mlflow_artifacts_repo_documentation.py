import re
from urllib.parse import urlparse, urlunparse
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.http_artifact_repo import HttpArtifactRepository
from mlflow.tracking._tracking_service.utils import get_tracking_uri
Scheme wrapper around HttpArtifactRepository for mlflow-artifacts server functionality