from unittest.mock import MagicMock, patch

from doc_forge.scribe.generate import REQUIRED_HEADINGS, DocGenerator


@patch("doc_forge.scribe.generate.ForgeRuntimeCoordinator", autospec=True)
@patch("requests.post")
def test_generate_success(mock_post, _mock_coordinator, mock_config, mock_llm_response):
    mock_post.return_value = mock_llm_response
    # Mock return needs to have valid headings
    valid_doc = "\\n".join(REQUIRED_HEADINGS) + "\\nContent."
    mock_llm_response.json.return_value = {"content": valid_doc}

    gen = DocGenerator(mock_config)
    result = gen.generate("test.py", "code", {})

    assert result.strip() == valid_doc.strip()
    # Should be called twice: Plan, Draft
    assert mock_post.call_count == 2


@patch("doc_forge.scribe.generate.ForgeRuntimeCoordinator", autospec=True)
@patch("requests.post")
def test_generate_retry(mock_post, _mock_coordinator, mock_config, mock_llm_response):
    # First draft fails validation (missing headings), second succeeds
    bad_draft = "# Wrong\\nBad content"
    good_draft = "\\n".join(REQUIRED_HEADINGS) + "\\nGood content"

    mock_post.side_effect = [
        MagicMock(status_code=200, json=lambda: {"content": "Plan"}),
        MagicMock(status_code=200, json=lambda: {"content": bad_draft}),
        MagicMock(status_code=200, json=lambda: {"content": good_draft}),
    ]

    gen = DocGenerator(mock_config)
    result = gen.generate("test.py", "code", {})

    assert "Good content" in result
    assert mock_post.call_count == 3


@patch("doc_forge.scribe.generate.ForgeRuntimeCoordinator", autospec=True)
@patch("requests.post")
def test_generate_accepts_openai_compat_completion(mock_post, _mock_coordinator, mock_config):
    valid_doc = "\n".join(REQUIRED_HEADINGS) + "\nCompat content."
    mock_post.side_effect = [
        MagicMock(status_code=200, json=lambda: {"choices": [{"text": "Plan"}]}),
        MagicMock(status_code=200, json=lambda: {"choices": [{"text": valid_doc}]}),
    ]

    gen = DocGenerator(mock_config)
    result = gen.generate("test.py", "code", {})

    assert "Compat content." in result


@patch("doc_forge.scribe.generate.ForgeRuntimeCoordinator", autospec=True)
@patch("requests.post")
def test_generate_uses_model_contract_when_managed_llm_disabled(mock_post, _mock_coordinator, mock_config):
    mock_config.enable_managed_llm = False
    valid_doc = "\n".join(REQUIRED_HEADINGS) + "\nContract content."

    class _ModelConfig:
        def __init__(self):
            self.calls = 0

        def generate_payload(self, *args, **kwargs):
            _ = args, kwargs
            self.calls += 1
            if self.calls == 1:
                return {"response": "Plan"}
            return {"response": valid_doc}

    with patch("doc_forge.scribe.generate.get_model_config", return_value=_ModelConfig()):
        gen = DocGenerator(mock_config)
        result = gen.generate("test.py", "code", {})

    assert "Contract content." in result
    assert mock_post.call_count == 0
