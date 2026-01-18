import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from forgeengine import engine as eng
from forgeengine.engine import NarrativeEngine


def test_engine_response(tmp_path, monkeypatch):
    def dummy_from_pretrained(name, *a, **k):
        return object()

    monkeypatch.setattr(eng, "AutoTokenizer", type("T", (), {"from_pretrained": dummy_from_pretrained}))
    monkeypatch.setattr(eng, "AutoModelForCausalLM", type("M", (), {"from_pretrained": dummy_from_pretrained}))
    monkeypatch.setattr(
        eng,
        "pipeline",
        lambda *args, **kwargs: lambda txt, **kw: [{"generated_text": txt.upper()}],
    )

    engine = NarrativeEngine(memory_path=str(tmp_path / "mem.json"), model_name="dummy")
    response = engine.respond("hello")
    assert "HELLO" in response


