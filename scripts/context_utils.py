import json
import logging
from pathlib import Path
from eidosian_core import eidosian

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_DIR = PROJECT_ROOT / "context"
CONFIG_PATH = CONTEXT_DIR / "config.json"


@eidosian()
def ensure_context_dir():
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)


@eidosian()
def load_config(path: Path = CONFIG_PATH):
    ensure_context_dir()
    try:
        with path.open() as fh:
            return json.load(fh)
    except FileNotFoundError as exc:
        raise SystemExit(f"config file missing: {path}: {exc}")


@eidosian()
def strip_ansi(text: str) -> str:
    # Remove ANSI escape sequences to keep JSON clean
    import re

    ansi_re = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    return ansi_re.sub("", text)


@eidosian()
def setup_logging(config: dict):
    logging_config = config.get("logging", {})
    level_name = logging_config.get("level", "INFO")
    log_format = logging_config.get(
        "format", "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    log_path = logging_config.get("path")
    logger = logging.getLogger("context")
    logger.setLevel(getattr(logging, level_name.upper(), logging.INFO))

    def _configure_handler(handler):
        handler.setLevel(logger.level)
        handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(handler)

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    _configure_handler(console_handler)
    if log_path:
        path = Path(log_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        _configure_handler(file_handler)
    return logger
