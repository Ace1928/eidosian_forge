from __future__ import annotations

import json
from pathlib import Path

from code_forge.eval_os.contracts import (
    ArtifactContract,
    EvalConfig,
    TaskSpec,
    write_eval_config_matrix,
    write_taskbank,
)
from code_forge.eval_os.runner import EvalRunOptions, run_eval_suite


def _build_taskbank(path: Path) -> None:
    tasks = [
        TaskSpec(
            task_id="eval.sample.ok",
            task_type="swe",
            description="simple success command",
            command="printf 'EVAL_OK\\n'",
            contract=ArtifactContract(
                require_zero_exit=True,
                stdout_must_contain=("EVAL_OK",),
                stderr_must_not_contain=("Traceback",),
            ),
        ),
        TaskSpec(
            task_id="eval.sample.file",
            task_type="docs",
            description="create a deterministic artifact file",
            command="mkdir -p tmp/eval_os && printf 'doc' > tmp/eval_os/out.txt",
            contract=ArtifactContract(
                require_zero_exit=True,
                required_paths=("tmp/eval_os/out.txt",),
            ),
        ),
    ]
    write_taskbank(path, tasks=tasks, metadata={"suite": "unit"})


def _build_matrix(path: Path) -> None:
    write_eval_config_matrix(
        path,
        configs=[
            EvalConfig(
                config_id="cortex_only",
                name="cortex_only",
                toggles={"CACHE_MODE": "off", "VALIDATORS": "minimal"},
            ),
            EvalConfig(
                config_id="forge_on",
                name="forge_on",
                toggles={"CACHE_MODE": "swr", "VALIDATORS": "all"},
            ),
        ],
        metadata={"suite": "unit"},
    )


def test_run_eval_suite_record_and_replay(tmp_path: Path) -> None:
    taskbank = tmp_path / "taskbank.json"
    matrix = tmp_path / "config_matrix.json"
    _build_taskbank(taskbank)
    _build_matrix(matrix)

    output_record = tmp_path / "record"
    payload_record = run_eval_suite(
        EvalRunOptions(
            taskbank_path=taskbank,
            config_matrix_path=matrix,
            output_dir=output_record,
            repo_root=tmp_path,
            repeats=1,
            replay_mode="record",
            max_parallel=2,
        )
    )
    assert payload_record["run_stats"]["run_count"] == 4
    assert payload_record["run_stats"]["success_rate"] == 1.0
    assert payload_record["run_stats"]["replay_hits"] == 0

    output_replay = tmp_path / "replay"
    payload_replay = run_eval_suite(
        EvalRunOptions(
            taskbank_path=taskbank,
            config_matrix_path=matrix,
            output_dir=output_replay,
            repo_root=tmp_path,
            repeats=1,
            replay_mode="replay",
            max_parallel=1,
            replay_store_path=output_record / "replay_store",
        )
    )
    assert payload_replay["run_stats"]["run_count"] == 4
    assert payload_replay["run_stats"]["replay_hits"] == 4
    assert payload_replay["run_stats"]["replay_misses"] == 0

    summary_path = output_replay / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "code_forge_eval_report_v1"
