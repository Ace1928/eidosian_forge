import re
import warnings
from pathlib import Path
from typing import Any, Dict, TypeVar
from urllib.parse import urlparse
from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.artifact_repository_registry import get_registered_artifact_repositories
from mlflow.utils.uri import is_local_uri
def register_artifact_dataset_sources():
    from mlflow.data.dataset_source_registry import register_dataset_source
    registered_source_schemes = set()
    artifact_schemes_to_exclude = ['http', 'https', 'runs', 'models', 'mlflow-artifacts', 'dbfs']
    schemes_to_artifact_repos = get_registered_artifact_repositories()
    for scheme, artifact_repo in schemes_to_artifact_repos.items():
        if scheme in artifact_schemes_to_exclude or scheme in registered_source_schemes:
            continue
        if 'ArtifactRepository' in artifact_repo.__name__:
            dataset_source_name = artifact_repo.__name__.replace('ArtifactRepository', 'ArtifactDatasetSource')
        else:
            scheme = str(scheme)

            def camelcase_scheme(scheme):
                parts = re.split('[-_]', scheme)
                return ''.join([part.capitalize() for part in parts])
            source_name_prefix = camelcase_scheme(scheme)
            dataset_source_name = source_name_prefix + 'ArtifactDatasetSource'
        try:
            registered_source_schemes.add(scheme)
            dataset_source = _create_dataset_source_for_artifact_repo(scheme=scheme, dataset_source_name=dataset_source_name)
            register_dataset_source(dataset_source)
        except Exception as e:
            warnings.warn(f"Failed to register a dataset source for URIs with scheme '{scheme}': {e}", stacklevel=2)