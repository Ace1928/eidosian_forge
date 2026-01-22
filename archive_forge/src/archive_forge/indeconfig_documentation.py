import os
from typing import Optional

    Configuration class containing all the settings for the application, sourced from environment variables.

    Attributes:
        JWT_SECRET_KEY (str): Secret key for JWT token generation and verification. Falls back to a default if not set.
        DATABASE_URI (Optional[str]): URI for the database connection. None if not set, indicating in-memory or default storage should be used.
        DEBUG_MODE (bool): Flag to indicate if the application should run in debug mode. Defaults to False.
    