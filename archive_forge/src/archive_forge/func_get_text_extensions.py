import os
import pathlib
from mimetypes import guess_type
def get_text_extensions():
    from mlflow.models.model import MLMODEL_FILE_NAME
    from mlflow.projects._project_spec import MLPROJECT_FILE_NAME
    return ['txt', 'log', 'err', 'cfg', 'conf', 'cnf', 'cf', 'ini', 'properties', 'prop', 'hocon', 'toml', 'yaml', 'yml', 'xml', 'json', 'js', 'py', 'py3', 'csv', 'tsv', 'md', 'rst', MLMODEL_FILE_NAME, MLPROJECT_FILE_NAME]