from typing import Any
from ray.util.annotations import PublicAPI
from ray.rllib.connectors.connector import Connector, ConnectorContext
@PublicAPI(stability='alpha')
def register_connector(name: str, cls: Connector):
    """Register a connector for use with RLlib.

    Args:
        name: Name to register.
        cls: Callable that creates an env.
    """
    if name in ALL_CONNECTORS:
        return
    if not issubclass(cls, Connector):
        raise TypeError('Can only register Connector type.', cls)
    ALL_CONNECTORS[name] = cls