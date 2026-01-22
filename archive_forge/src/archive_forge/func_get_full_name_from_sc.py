from typing import List, Optional
from mlflow.entities.model_registry import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_uc_registry_messages_pb2 import ModelVersion as ProtoModelVersion
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
from mlflow.protos.databricks_uc_registry_messages_pb2 import TemporaryCredentials
from mlflow.store.artifact.artifact_repo import ArtifactRepository
def get_full_name_from_sc(name, spark) -> str:
    """
    Constructs the full name of a registered model using the active catalog and schema in a spark
    session / context.

    Args:
        name: The model name provided by the user.
        spark: The active spark session.
    """
    num_levels = len(name.split('.'))
    if num_levels >= 3 or spark is None:
        return name
    catalog = spark.sql(_ACTIVE_CATALOG_QUERY).collect()[0]['catalog']
    if catalog in {'spark_catalog', 'hive_metastore'}:
        return name
    if num_levels == 2:
        return f'{catalog}.{name}'
    schema = spark.sql(_ACTIVE_SCHEMA_QUERY).collect()[0]['schema']
    return f'{catalog}.{schema}.{name}'