import logging
import os
import time
from contextlib import contextmanager
import sqlalchemy
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import sql
from sqlalchemy.pool import (
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INTERNAL_ERROR, TEMPORARILY_UNAVAILABLE
from mlflow.store.db.db_types import SQLITE
from mlflow.store.model_registry.dbmodels.models import (
from mlflow.store.tracking.dbmodels.initial_models import Base as InitialBase
from mlflow.store.tracking.dbmodels.models import (
def create_sqlalchemy_engine(db_uri):
    pool_size = MLFLOW_SQLALCHEMYSTORE_POOL_SIZE.get()
    pool_max_overflow = MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW.get()
    pool_recycle = MLFLOW_SQLALCHEMYSTORE_POOL_RECYCLE.get()
    echo = MLFLOW_SQLALCHEMYSTORE_ECHO.get()
    poolclass = MLFLOW_SQLALCHEMYSTORE_POOLCLASS.get()
    pool_kwargs = {}
    if pool_size:
        pool_kwargs['pool_size'] = pool_size
    if pool_max_overflow:
        pool_kwargs['max_overflow'] = pool_max_overflow
    if pool_recycle:
        pool_kwargs['pool_recycle'] = pool_recycle
    if echo:
        pool_kwargs['echo'] = echo
    if poolclass:
        pool_class_map = {'AssertionPool': AssertionPool, 'AsyncAdaptedQueuePool': AsyncAdaptedQueuePool, 'FallbackAsyncAdaptedQueuePool': FallbackAsyncAdaptedQueuePool, 'NullPool': NullPool, 'QueuePool': QueuePool, 'SingletonThreadPool': SingletonThreadPool, 'StaticPool': StaticPool}
        if poolclass not in pool_class_map:
            list_str = ' '.join(pool_class_map.keys())
            err_str = f'Invalid poolclass parameter: {poolclass}. Set environment variable poolclass to empty or one of the following values: {list_str}'
            _logger.warning(err_str)
            raise ValueError(err_str)
        pool_kwargs['poolclass'] = pool_class_map[poolclass]
    if pool_kwargs:
        _logger.info('Create SQLAlchemy engine with pool options %s', pool_kwargs)
    return sqlalchemy.create_engine(db_uri, pool_pre_ping=True, **pool_kwargs)