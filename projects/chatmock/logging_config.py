import logging
import logging.handlers
import os
import sys
from pathlib import Path

def setup_logging(verbose: bool = False, log_to_file: bool = True, log_dir: str | Path | None = None) -> None:
    """
    Configures logging for the ChatMock application.
    
    Args:
        verbose: If True, set console level to DEBUG, else INFO.
        log_to_file: If True, enables file logging.
        log_dir: Directory to store log files. Defaults to ~/.chatmock/logs.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture everything, handlers decide what to show
    
    # Remove existing handlers to avoid duplicates on reload/reconfiguration
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_level = logging.DEBUG if verbose else logging.INFO
    console_handler.setLevel(console_level)
    
    # Custom formatter for console (cleaner, less metadata for standard users)
    if verbose:
        console_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        console_fmt = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    # File Handler
    if log_to_file:
        if log_dir is None:
            home = Path.home()
            log_dir = home / ".chatmock" / "logs"
        else:
            log_dir = Path(log_dir)
        
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "chatmock.log"
            
            # Rotating file handler (10MB per file, max 5 backups)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_fmt)
            root_logger.addHandler(file_handler)
        except Exception as e:
            # Fallback if we can't write to the log file
            print(f"Warning: Failed to setup file logging: {e}", file=sys.stderr)

def get_logger(name: str) -> logging.Logger:
    """Returns a logger with the given name."""
    return logging.getLogger(name)
