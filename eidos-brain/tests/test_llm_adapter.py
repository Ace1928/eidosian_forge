from core.llm_adapter import LLMAdapter


def test_fallback_summary(monkeypatch) -> None:
    adapter = LLMAdapter()

    # ensure openai path is None so fallback triggers
    monkeypatch.setattr("core.llm_adapter.openai", None)

    long_text = "A" * 120
    summary = adapter.summarize(long_text)
    assert summary.endswith("...")
    assert len(summary) <= 100
