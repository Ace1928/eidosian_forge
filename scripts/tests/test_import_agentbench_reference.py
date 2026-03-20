from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "import_agentbench_reference.py"
    spec = importlib.util.spec_from_file_location("import_agentbench_reference", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aggregate_agentbench_reads_leaderboard_csv(tmp_path: Path) -> None:
    module = _load_script_module()
    csv_path = tmp_path / "leaderboard.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Email agentbench_fc@googlegroups.com to submit your result!,,,,,,,,,,,,,,",
                "Model,Model Organization,Result Source,Release Date,Success Rate (pass@1),,,,,,,,,,",
                ",,,,ALFWorld,,DB,,KG,,OS,,WebShop,,AVG",
                "Model A,Org A,AgentRL,10/2025,90.0,,70.0,,60.0,,50.0,,40.0,,62.0",
                "Claude Sonnet 4.5,Anthropic,-,11/2025,82.5,,71.2,,64.1,,38.0,,38.6,,58.9",
            ]
        ),
        encoding="utf-8",
    )

    payload = module.aggregate_agentbench(csv_path)

    assert payload["suite"] == "agentbench"
    assert payload["score"] == 0.62
    assert payload["metrics"]["rows_total"] == 2
    assert payload["leaderboard"]["best_model"] == "Model A"
    assert payload["leaderboard"]["best_open_model"] == "Model A"
