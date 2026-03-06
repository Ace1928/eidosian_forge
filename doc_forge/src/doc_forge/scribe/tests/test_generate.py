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
