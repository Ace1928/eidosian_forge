from urllib.parse import urlparse
from mlflow.store.artifact.optimized_s3_artifact_repo import OptimizedS3ArtifactRepository
from mlflow.store.artifact.s3_artifact_repo import _get_s3_client
def parse_s3_compliant_uri(self, uri):
    parsed = urlparse(uri)
    if parsed.scheme != 'r2':
        raise Exception(f'Not an R2 URI: {uri}')
    host = parsed.netloc
    path = parsed.path
    bucket = host.split('@')[0]
    if path.startswith('/'):
        path = path[1:]
    return (bucket, path)