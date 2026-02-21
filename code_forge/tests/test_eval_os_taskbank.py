from __future__ import annotations

import json
from pathlib import Path

import pytest
from code_forge.eval_os.contracts import (
    EvalConfig,
    TaskSpec,
    load_eval_config_matrix,
    load_taskbank,
    write_eval_config_matrix,
    write_taskbank,
)
from code_forge.eval_os.taskbank import (
    create_sample_config_matrix,
    create_sample_taskbank,
)


def test_create_sample_taskbank_and_matrix(tmp_path: Path) -> None:
    taskbank_path = tmp_path / "taskbank.json"
    matrix_path = tmp_path / "config_matrix.json"

    taskbank = create_sample_taskbank(taskbank_path)
    matrix = create_sample_config_matrix(matrix_path)

    assert taskbank_path.exists()
    assert matrix_path.exists()
    assert len(taskbank.get("tasks") or []) >= 1
    assert len(matrix.get("configs") or []) >= 1

    _, tasks, _ = load_taskbank(taskbank_path)
    loaded_matrix = load_eval_config_matrix(matrix_path)
    assert len(tasks) == len(taskbank["tasks"])
    assert len(loaded_matrix.configs) == len(matrix["configs"])


def test_write_taskbank_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_taskbank(tmp_path / "taskbank.json", tasks=[])


def test_load_taskbank_rejects_duplicate_task_ids(tmp_path: Path) -> None:
    path = tmp_path / "taskbank.json"
    payload = {
        "schema_version": "code_forge_taskbank_v1",
        "tasks": [
            TaskSpec(
                task_id="dup",
                task_type="swe",
                description="a",
                command="echo a",
            ).to_dict(),
            TaskSpec(
                task_id="dup",
                task_type="docs",
                description="b",
                command="echo b",
            ).to_dict(),
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_taskbank(path)


def test_eval_config_matrix_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "matrix.json"
    write_eval_config_matrix(
        path,
        configs=[
            EvalConfig(
                config_id="cfg1",
                name="cfg1",
                toggles={"CACHE_MODE": "off"},
            ),
            EvalConfig(
                config_id="cfg2",
                name="cfg2",
                toggles={"CACHE_MODE": "swr"},
            ),
        ],
    )
    matrix = load_eval_config_matrix(path)
    assert [cfg.config_id for cfg in matrix.configs] == ["cfg1", "cfg2"]
