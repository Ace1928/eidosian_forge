import enum
import functools
import redis
from redis import exceptions as redis_exceptions
Checks if a client is attached to a new enough redis server.