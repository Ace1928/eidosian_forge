from __future__ import annotations
import logging
import sys
from typing import Final
def setup_formatter(logger: logging.Logger) -> None:
    """Set up the console formatter for a given logger."""
    if hasattr(logger, 'streamlit_console_handler'):
        logger.removeHandler(logger.streamlit_console_handler)
    logger.streamlit_console_handler = logging.StreamHandler()
    from streamlit import config
    if config._config_options:
        message_format = config.get_option('logger.messageFormat')
    else:
        message_format = DEFAULT_LOG_MESSAGE
    formatter = logging.Formatter(fmt=message_format)
    formatter.default_msec_format = '%s.%03d'
    logger.streamlit_console_handler.setFormatter(formatter)
    logger.addHandler(logger.streamlit_console_handler)