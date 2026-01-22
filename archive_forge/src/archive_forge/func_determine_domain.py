from typing import Any, Optional
def determine_domain(client_universe_domain: Optional[str], universe_domain_env: Optional[str]) -> str:
    """Return the universe domain used by the client.

    Args:
        client_universe_domain (Optional[str]): The universe domain configured via the client options.
        universe_domain_env (Optional[str]): The universe domain configured via the
        "GOOGLE_CLOUD_UNIVERSE_DOMAIN" environment variable.

    Returns:
        str: The universe domain to be used by the client.

    Raises:
        ValueError: If the universe domain is an empty string.
    """
    universe_domain = DEFAULT_UNIVERSE
    if client_universe_domain is not None:
        universe_domain = client_universe_domain
    elif universe_domain_env is not None:
        universe_domain = universe_domain_env
    if len(universe_domain.strip()) == 0:
        raise EmptyUniverseError
    return universe_domain