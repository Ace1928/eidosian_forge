from __future__ import annotations
from typing import TYPE_CHECKING, Iterator, List, Optional
from langchain_core.utils import get_from_env
Convenience constructor that builds the odsp.ODPS MaxCompute client from
            given parameters.

        Args:
            endpoint: MaxCompute endpoint.
            project: A project is a basic organizational unit of MaxCompute, which is
                similar to a database.
            access_id: MaxCompute access ID. Should be passed in directly or set as the
                environment variable `MAX_COMPUTE_ACCESS_ID`.
            secret_access_key: MaxCompute secret access key. Should be passed in
                directly or set as the environment variable
                `MAX_COMPUTE_SECRET_ACCESS_KEY`.
        