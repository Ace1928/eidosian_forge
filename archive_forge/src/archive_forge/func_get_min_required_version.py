import os
import pathlib
import grpc_tools  # type: ignore
from grpc_tools import protoc  # type: ignore
import importlib.metadata
from packaging import version
def get_min_required_version(requirements_file_name: str, package_name: str) -> str:
    with open(requirements_file_name) as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.strip().split('>=')
            if tokens[0] == package_name:
                if len(tokens) == 2:
                    return tokens[1]
                else:
                    raise ValueError(f'Minimum version not specified for package `{package_name}`')
    raise ValueError(f'Package `{package_name}` not found in requirements file')