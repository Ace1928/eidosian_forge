import os
from pathlib import Path
from doc_forge.scribe.config import ScribeConfig

def test_config_defaults(temp_forge_root):
    cfg = ScribeConfig.from_env(forge_root=temp_forge_root)
    assert cfg.forge_root == temp_forge_root
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8930
    assert cfg.max_file_bytes == 4 * 1024 * 1024
    assert ".py" in cfg.include_suffixes

def test_config_env_override(temp_forge_root):
    os.environ["EIDOS_DOC_FORGE_HOST"] = "0.0.0.0"
    os.environ["EIDOS_DOC_FORGE_PORT"] = "9000"
    try:
        cfg = ScribeConfig.from_env(forge_root=temp_forge_root)
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 9000
    finally:
        del os.environ["EIDOS_DOC_FORGE_HOST"]
        del os.environ["EIDOS_DOC_FORGE_PORT"]

def test_path_resolution(temp_forge_root):
    cfg = ScribeConfig.from_env(forge_root=temp_forge_root)
    assert cfg.runtime_root == temp_forge_root / "doc_forge" / "runtime"
    assert cfg.staging_root == cfg.runtime_root / "staging_docs"
