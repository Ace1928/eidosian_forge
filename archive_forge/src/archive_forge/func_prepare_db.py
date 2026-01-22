import os
import time
import asyncio
import contextlib
from lazyops.imports._sqlalchemy import require_sql
from sqlalchemy import create_engine, event, exc
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.orm import Session, scoped_session
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine, async_scoped_session
from lazyops.utils.logs import logger
from lazyops.utils import Json
from lazyops.types import BaseModel, lazyproperty, BaseSettings, Field
from typing import Any, Generator, AsyncGenerator, Optional, Union, Type, Dict, cast, TYPE_CHECKING, List, Tuple, TypeVar, Callable
from pydantic.networks import PostgresDsn
from lazyops.libs.psqldb.retry import reconnecting_engine
from lazyops.utils.helpers import import_string
@classmethod
def prepare_db(cls, role_user: Optional[str]=None, role_password: Optional[str]=None, role_createdb: Optional[bool]=True, role_createrole: Optional[bool]=True, role_replication: Optional[bool]=True, role_bypassrls: Optional[bool]=True, role_login: Optional[bool]=None, role_options: Optional[Dict[str, Any]]=None, role_superuser: Optional[bool]=True, db_name: Optional[str]=None, db_owner: Optional[str]=None, db_grants: Optional[List[Tuple[str, Optional['Privilege']]]]=None, db_options: Optional[Dict[str, Any]]=None, db_admin_uri: Optional[Union[str, PostgresDsn]]=None, statements: Optional[List[str]]=None, grant_public_schema: Optional[bool]=True, verbose: Optional[bool]=True, raise_exception: Optional[bool]=False, **kwargs):
    """
        Runs initialization for the DB
        """
    from lazyops.libs.dbinit import Controller, GrantTo, Privilege, Database, Role
    pg_user = role_user or cls.ctx.uri.user
    pg_pass = role_password or cls.ctx.uri.password
    if role_login is None:
        role_login = bool(pg_pass)
    if not role_options:
        role_options = {}
    if not db_options:
        db_options = {}
    if pg_user == 'postgres':
        role = Role(name=pg_user)
    else:
        role = Role(name=pg_user, createdb=role_createdb, createrole=role_createrole, replication=role_replication, bypassrls=role_bypassrls, login=role_login, password=pg_pass, superuser=role_superuser, **role_options)
    if verbose:
        logger.info(f'Creating Role {pg_user}: {role}')
    pg_db = db_name or cls.ctx.uri.path[1:]
    db_owner = role if db_owner is None else Role(name=db_owner)
    if not db_grants:
        db_grants = [GrantTo(privileges=[Privilege.ALL_PRIVILEGES], to=[role])]
    db = Database(name=pg_db, owner=db_owner, grants=db_grants, **db_options)
    if verbose:
        logger.info(f'Creating Database {pg_db}: {db}')
    db_admin_uri = db_admin_uri or cls.admin_uri
    try:
        Controller.run_all(engine=create_engine(url=db_admin_uri))
    except Exception as e:
        logger.trace(f'Error creating database: {pg_db}', error=e)
        if raise_exception:
            raise e
    engine = create_engine(url=cls.get_admin_uri(db=pg_db))
    if statements:
        for statement in statements:
            if verbose:
                logger.info(f'Executing statement: {statement}')
            try:
                engine.execute(statement=statement)
            except Exception as e:
                logger.trace(f'Error executing statement: {statement}', error=e)
                if raise_exception:
                    raise e
    if grant_public_schema:
        pg_user_encoded = f'"{pg_user}"' if '_' in pg_user or '-' in pg_user else pg_user
        schema_statements = [f'GRANT ALL PRIVILEGES ON SCHEMA public TO {pg_user_encoded}', f'ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL PRIVILEGES ON TABLES TO {pg_user_encoded}', f'GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO {pg_user_encoded}', f'GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO {pg_user_encoded}']
        if statements:
            schema_statements = [statement for statement in schema_statements if statement not in statements]
        for statement in schema_statements:
            if verbose:
                logger.info(f'Executing statement: {statement}')
            try:
                engine.execute(statement=statement)
            except Exception as e:
                logger.trace(f'Error executing statement: {statement}', error=e)
                if raise_exception:
                    raise e
    if verbose:
        logger.info('Completed DB initialization')