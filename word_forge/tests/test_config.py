"""Tests for :mod:`word_forge.config`."""

import importlib
import os
import sys
from pathlib import Path


def test_get_full_path_joins_data_dir(tmp_path: Path) -> None:
    """Verify :func:`Config.get_full_path` joins ``parser.data_dir`` with a relative path."""

    repo_src = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(repo_src))

    os.environ["WORD_FORGE_DATA_DIR"] = str(tmp_path)

    import word_forge.config as cfg

    importlib.reload(cfg)
    result = cfg.config.get_full_path("example.txt")
    assert result == Path(tmp_path) / "example.txt"
