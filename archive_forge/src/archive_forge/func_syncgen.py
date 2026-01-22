import pytest
from .. import aclosing, async_generator, yield_, asynccontextmanager
@asynccontextmanager
def syncgen():
    yield