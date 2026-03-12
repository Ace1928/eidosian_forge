from __future__ import annotations

from pathlib import Path

from eidosian_runtime.capabilities import collect_runtime_capabilities, write_runtime_capabilities


def test_collect_runtime_capabilities_has_contract(monkeypatch, tmp_path: Path) -> None:
    forge_root = tmp_path / "forge"
    forge_root.mkdir(parents=True)
    monkeypatch.setenv("EIDOS_FORGE_ROOT", str(forge_root))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PREFIX", str(tmp_path / "usr"))
    caps = collect_runtime_capabilities()
    assert caps.contract == "eidos.runtime_capabilities.v1"
    assert caps.forge_root == str(forge_root.resolve())


def test_write_runtime_capabilities_writes_json(monkeypatch, tmp_path: Path) -> None:
    forge_root = tmp_path / "forge"
    forge_root.mkdir(parents=True)
    output = forge_root / "data" / "runtime" / "platform_capabilities.json"
    monkeypatch.setenv("EIDOS_FORGE_ROOT", str(forge_root))
    payload = write_runtime_capabilities(output)
    assert output.exists()
    assert payload["contract"] == "eidos.runtime_capabilities.v1"
