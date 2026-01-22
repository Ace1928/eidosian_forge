import typing
from redis.backoff import (
def default_backoff() -> typing.Type[AbstractBackoff]:
    return EqualJitterBackoff()