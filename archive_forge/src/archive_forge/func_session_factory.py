import time
from kombu.utils.compat import register_after_fork
from sqlalchemy import create_engine
from sqlalchemy.exc import DatabaseError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from celery.utils.time import get_exponential_backoff_interval
def session_factory(self, dburi, **kwargs):
    engine, session = self.create_session(dburi, **kwargs)
    self.prepare_models(engine)
    return session()