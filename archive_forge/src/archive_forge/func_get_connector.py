from typing import Any
from ray.util.annotations import PublicAPI
from ray.rllib.connectors.connector import Connector, ConnectorContext
@PublicAPI(stability='alpha')
def get_connector(name: str, ctx: ConnectorContext, params: Any=None) -> Connector:
    """Get a connector by its name and serialized config.

    Args:
        name: name of the connector.
        ctx: Connector context.
        params: serialized parameters of the connector.

    Returns:
        Constructed connector.
    """
    if name not in ALL_CONNECTORS:
        raise NameError('connector not found.', name)
    return ALL_CONNECTORS[name].from_state(ctx, params)