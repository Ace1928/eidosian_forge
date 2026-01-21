import pytest
from pathlib import Path
from gis_forge import GisCore
from pydantic import BaseModel

class ServerConfig(BaseModel):
    host: str
    port: int

def test_gis_basic(tmp_path):
    f = tmp_path / "config.json"
    gis = GisCore(persistence_path=f)
    gis.set("foo.bar", "baz")
    assert gis.get("foo.bar") == "baz"
    assert f.exists()

def test_gis_yaml(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("server:\n  host: localhost\n  port: 8080", encoding="utf-8")
    
    gis = GisCore(persistence_path=f)
    assert gis.get("server.host") == "localhost"
    assert gis.get("server.port") == 8080

def test_gis_validation(tmp_path):
    gis = GisCore()
    gis.set("server.host", "127.0.0.1")
    gis.set("server.port", 9000)
    
    model = gis.validate_config(ServerConfig, "server")
    assert model.port == 9000

def test_gis_env_override(monkeypatch):
    gis = GisCore()
    gis.set("app.debug", False)
    
    monkeypatch.setenv("EIDOS_APP_DEBUG", "true")
    assert gis.get("app.debug") is True  # JSON parsing of "true"