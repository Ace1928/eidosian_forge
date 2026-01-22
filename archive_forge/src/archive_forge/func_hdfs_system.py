import os
import posixpath
import tempfile
import urllib.parse
from contextlib import contextmanager
import packaging.version
from mlflow.entities import FileInfo
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils.file_utils import mkdir, relative_path_to_artifact_path
@contextmanager
def hdfs_system(scheme, host, port):
    """
    hdfs system context - Attempt to establish the connection to hdfs
    and yields HadoopFileSystem

    Args:
        scheme: scheme or use hdfs:// as default
        host: hostname or when relaying on the core-site.xml config use 'default'
        port: port or when relaying on the core-site.xml config use 0
    """
    import pyarrow
    kerb_ticket = MLFLOW_KERBEROS_TICKET_CACHE.get()
    kerberos_user = MLFLOW_KERBEROS_USER.get()
    extra_conf = _parse_extra_conf(MLFLOW_PYARROW_EXTRA_CONF.get())
    host = scheme + '://' + host if host else 'default'
    if packaging.version.parse(pyarrow.__version__) < packaging.version.parse('2.0.0'):
        connected = pyarrow.fs.HadoopFileSystem(host=host, port=port or 0, user=kerberos_user, kerb_ticket=kerb_ticket, extra_conf=extra_conf)
    else:
        connected = pyarrow.hdfs.connect(host=host, port=port or 0, user=kerberos_user, kerb_ticket=kerb_ticket, extra_conf=extra_conf)
    yield connected
    connected.close()