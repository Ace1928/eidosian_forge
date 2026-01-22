from __future__ import annotations
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Union
import sqlalchemy
from langchain_core._api import deprecated
from langchain_core.utils import get_from_env
from sqlalchemy import (
from sqlalchemy.engine import Engine, Result
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from sqlalchemy.schema import CreateTable
from sqlalchemy.sql.expression import Executable
from sqlalchemy.types import NullType
@classmethod
def from_databricks(cls, catalog: str, schema: str, host: Optional[str]=None, api_token: Optional[str]=None, warehouse_id: Optional[str]=None, cluster_id: Optional[str]=None, engine_args: Optional[dict]=None, **kwargs: Any) -> SQLDatabase:
    """
        Class method to create an SQLDatabase instance from a Databricks connection.
        This method requires the 'databricks-sql-connector' package. If not installed,
        it can be added using `pip install databricks-sql-connector`.

        Args:
            catalog (str): The catalog name in the Databricks database.
            schema (str): The schema name in the catalog.
            host (Optional[str]): The Databricks workspace hostname, excluding
                'https://' part. If not provided, it attempts to fetch from the
                environment variable 'DATABRICKS_HOST'. If still unavailable and if
                running in a Databricks notebook, it defaults to the current workspace
                hostname. Defaults to None.
            api_token (Optional[str]): The Databricks personal access token for
                accessing the Databricks SQL warehouse or the cluster. If not provided,
                it attempts to fetch from 'DATABRICKS_TOKEN'. If still unavailable
                and running in a Databricks notebook, a temporary token for the current
                user is generated. Defaults to None.
            warehouse_id (Optional[str]): The warehouse ID in the Databricks SQL. If
                provided, the method configures the connection to use this warehouse.
                Cannot be used with 'cluster_id'. Defaults to None.
            cluster_id (Optional[str]): The cluster ID in the Databricks Runtime. If
                provided, the method configures the connection to use this cluster.
                Cannot be used with 'warehouse_id'. If running in a Databricks notebook
                and both 'warehouse_id' and 'cluster_id' are None, it uses the ID of the
                cluster the notebook is attached to. Defaults to None.
            engine_args (Optional[dict]): The arguments to be used when connecting
                Databricks. Defaults to None.
            **kwargs (Any): Additional keyword arguments for the `from_uri` method.

        Returns:
            SQLDatabase: An instance of SQLDatabase configured with the provided
                Databricks connection details.

        Raises:
            ValueError: If 'databricks-sql-connector' is not found, or if both
                'warehouse_id' and 'cluster_id' are provided, or if neither
                'warehouse_id' nor 'cluster_id' are provided and it's not executing
                inside a Databricks notebook.
        """
    try:
        from databricks import sql
    except ImportError:
        raise ValueError('databricks-sql-connector package not found, please install with `pip install databricks-sql-connector`')
    context = None
    try:
        from dbruntime.databricks_repl_context import get_context
        context = get_context()
    except ImportError:
        pass
    default_host = context.browserHostName if context else None
    if host is None:
        host = get_from_env('host', 'DATABRICKS_HOST', default_host)
    default_api_token = context.apiToken if context else None
    if api_token is None:
        api_token = get_from_env('api_token', 'DATABRICKS_TOKEN', default_api_token)
    if warehouse_id is None and cluster_id is None:
        if context:
            cluster_id = context.clusterId
        else:
            raise ValueError("Need to provide either 'warehouse_id' or 'cluster_id'.")
    if warehouse_id and cluster_id:
        raise ValueError("Can't have both 'warehouse_id' or 'cluster_id'.")
    if warehouse_id:
        http_path = f'/sql/1.0/warehouses/{warehouse_id}'
    else:
        http_path = f'/sql/protocolv1/o/0/{cluster_id}'
    uri = f'databricks://token:{api_token}@{host}?http_path={http_path}&catalog={catalog}&schema={schema}'
    return cls.from_uri(database_uri=uri, engine_args=engine_args, **kwargs)