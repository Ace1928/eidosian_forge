import logging
import click
from packaging.requirements import InvalidRequirement, Requirement
from mlflow.models import python_api
from mlflow.models.flavor_backend_registry import get_flavor_backend
from mlflow.models.model import update_model_requirements
from mlflow.utils import cli_args
from mlflow.utils import env_manager as _EnvManager
@commands.command('build-docker')
@cli_args.MODEL_URI_BUILD_DOCKER
@click.option('--name', '-n', default='mlflow-pyfunc-servable', help='Name to use for built image')
@cli_args.ENV_MANAGER
@cli_args.MLFLOW_HOME
@cli_args.INSTALL_JAVA
@cli_args.INSTALL_MLFLOW
@cli_args.ENABLE_MLSERVER
def build_docker(**kwargs):
    """
    Builds a Docker image whose default entrypoint serves an MLflow model at port 8080, using the
    python_function flavor. The container serves the model referenced by ``--model-uri``, if
    specified when ``build-docker`` is called. If ``--model-uri`` is not specified when build_docker
    is called, an MLflow Model directory must be mounted as a volume into the /opt/ml/model
    directory in the container.

    Building a Docker image with ``--model-uri``:

    .. code:: bash

        # Build a Docker image named 'my-image-name' that serves the model from run 'some-run-uuid'
        # at run-relative artifact path 'my-model'
        mlflow models build-docker --model-uri "runs:/some-run-uuid/my-model" --name "my-image-name"
        # Serve the model
        docker run -p 5001:8080 "my-image-name"

    Building a Docker image without ``--model-uri``:

    .. code:: bash

        # Build a generic Docker image named 'my-image-name'
        mlflow models build-docker --name "my-image-name"
        # Mount the model stored in '/local/path/to/artifacts/model' and serve it
        docker run --rm -p 5001:8080 -v /local/path/to/artifacts/model:/opt/ml/model "my-image-name"

    .. important::

        Since MLflow 2.10.1, the Docker image built with ``--model-uri`` does **not install Java**
        for improved performance, unless the model flavor is one of ``["johnsnowlabs", "h2o",
        "mleap", "spark"]``. If you need to install Java for other flavors, e.g. custom Python model
        that uses SparkML, please specify the ``--install-java`` flag to enforce Java installation.

    .. warning::

        The image built without ``--model-uri`` doesn't support serving models with RFunc / Java
        MLeap model server.

    NB: by default, the container will start nginx and gunicorn processes. If you don't need the
    nginx process to be started (for instance if you deploy your container to Google Cloud Run),
    you can disable it via the DISABLE_NGINX environment variable:

    .. code:: bash

        docker run -p 5001:8080 -e DISABLE_NGINX=true "my-image-name"

    See https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html for more information on the
    'python_function' flavor.
    """
    python_api.build_docker(**kwargs)