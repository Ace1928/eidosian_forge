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
def from_cnosdb(cls, url: str='127.0.0.1:8902', user: str='root', password: str='', tenant: str='cnosdb', database: str='public') -> SQLDatabase:
    """
        Class method to create an SQLDatabase instance from a CnosDB connection.
        This method requires the 'cnos-connector' package. If not installed, it
        can be added using `pip install cnos-connector`.

        Args:
            url (str): The HTTP connection host name and port number of the CnosDB
                service, excluding "http://" or "https://", with a default value
                of "127.0.0.1:8902".
            user (str): The username used to connect to the CnosDB service, with a
                default value of "root".
            password (str): The password of the user connecting to the CnosDB service,
                with a default value of "".
            tenant (str): The name of the tenant used to connect to the CnosDB service,
                with a default value of "cnosdb".
            database (str): The name of the database in the CnosDB tenant.

        Returns:
            SQLDatabase: An instance of SQLDatabase configured with the provided
            CnosDB connection details.
        """
    try:
        from cnosdb_connector import make_cnosdb_langchain_uri
        uri = make_cnosdb_langchain_uri(url, user, password, tenant, database)
        return cls.from_uri(database_uri=uri)
    except ImportError:
        raise ValueError('cnos-connector package not found, please install with `pip install cnos-connector`')