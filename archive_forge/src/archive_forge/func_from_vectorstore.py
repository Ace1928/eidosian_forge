from __future__ import annotations
from typing import Any, Tuple
from langchain_community.vectorstores.redis import Redis
from langchain_community.vectorstores.redis.filters import (
from langchain_community.vectorstores.redis.schema import RedisModel
from langchain.chains.query_constructor.ir import (
@classmethod
def from_vectorstore(cls, vectorstore: Redis) -> RedisTranslator:
    return cls(vectorstore._schema)