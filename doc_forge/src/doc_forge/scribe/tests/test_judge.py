import pytest
from unittest.mock import patch, MagicMock
from doc_forge.scribe.judge import FederatedJudge, REQUIRED_HEADINGS

def test_heuristics_perfect(mock_config):
    judge = FederatedJudge(mock_config)
    markdown = "\\n".join(REQUIRED_HEADINGS) + "\\n- Bullet\\n`code`\\n1. Number"
    source = "def foo(): pass"
    
    scorecard = judge.evaluate(markdown=markdown, source_text=source, rel_path="test.py", metadata={})
    assert scorecard["evaluated_at"] != "now"
    
    # Check individual judges
    structure = next(j for j in scorecard["judges"] if j["name"] == "structure_contract")
    assert structure["score"] == 1.0
    
    safety = next(j for j in scorecard["judges"] if j["name"] == "anti_placeholder")
    assert safety["score"] == 1.0

def test_heuristics_bad(mock_config):
    judge = FederatedJudge(mock_config)
    markdown = "# Bad Doc\\nTODO: write this"
    source = "def foo(): pass"
    
    scorecard = judge.evaluate(markdown=markdown, source_text=source, rel_path="test.py", metadata={})
    
    safety = next(j for j in scorecard["judges"] if j["name"] == "anti_placeholder")
    assert safety["score"] < 1.0 # Penalty for TODO

@patch("requests.post")
def test_llm_judge(mock_post, mock_config):
    mock_config.dry_run = False
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"content": "0.85"}
    
    judge = FederatedJudge(mock_config)
    markdown = "\\n".join(REQUIRED_HEADINGS)
    source = "code"
    
    scorecard = judge.evaluate(markdown=markdown, source_text=source, rel_path="test.py", metadata={})
    
    llm = next(j for j in scorecard["judges"] if j["name"] == "llm_quality")
    assert llm["score"] == 0.85


@patch("requests.post", side_effect=RuntimeError("network down"))
def test_llm_judge_error_details(mock_post, mock_config):
    mock_config.dry_run = False
    judge = FederatedJudge(mock_config)
    markdown = "\\n".join(REQUIRED_HEADINGS)
    source = "code"
    scorecard = judge.evaluate(markdown=markdown, source_text=source, rel_path="test.py", metadata={})
    llm = next(j for j in scorecard["judges"] if j["name"] == "llm_quality")
    assert llm["score"] == 0.5
    assert "error" in llm["details"]
