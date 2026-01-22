import logging
import os
import re
import subprocess
import sys
from mlflow.exceptions import MlflowException
from mlflow.models import FlavorBackend
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.string_utils import quote
def build_image(self, model_uri, image_name, install_mlflow, mlflow_home, enable_mlserver):
    pass