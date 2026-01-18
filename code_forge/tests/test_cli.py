import sys, pathlib, json
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from forgeengine.cli import load_config, discover_local_models


def test_load_config_file(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"memory": "x.json", "think": 1, "model": "m", "max_tokens": 5}))
    data = load_config(str(cfg))
    assert data["memory"] == "x.json"
    assert data["think"] == 1
    assert "adapter" not in data


def test_discover_local_models(tmp_path):
    assert discover_local_models([str(tmp_path)]) == []
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    assert discover_local_models([str(tmp_path)]) == [str(model_dir)]


