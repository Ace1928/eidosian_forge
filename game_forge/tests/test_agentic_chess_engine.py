from __future__ import annotations

from typing import Any

import pytest

from agentic_chess.agents import RandomAgent
from agentic_chess.cli import DependencyError, main
from agentic_chess.engine import MatchConfig, play_match


def test_cli_dependency_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise DependencyError("python-chess is required")

    monkeypatch.setattr("agentic_chess.cli.play_match", _raise)
    code = main(["--white", "random", "--black", "random", "--max-moves", "1"])

    captured = capsys.readouterr()
    assert code == 1
    assert "ERROR python-chess is required" in captured.err


def test_play_match_random_agents() -> None:
    pytest.importorskip("chess", reason="python-chess required")
    white = RandomAgent()
    black = RandomAgent()
    result = play_match(white, black, MatchConfig(max_moves=6, seed=0))

    assert result.result in {"1-0", "0-1", "1/2-1/2"}
    assert result.termination
    assert len(result.moves) <= 6


def test_cli_pgn_output(tmp_path: Any) -> None:
    pytest.importorskip("chess", reason="python-chess required")
    output = tmp_path / "match.pgn"
    code = main(
        [
            "--white",
            "random",
            "--black",
            "random",
            "--max-moves",
            "2",
            "--pgn-output",
            str(output),
        ]
    )

    assert code == 0
    assert output.exists()
    assert output.read_text(encoding="utf-8").strip()
