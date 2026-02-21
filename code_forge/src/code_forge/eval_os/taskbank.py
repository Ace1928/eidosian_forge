from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from code_forge.eval_os.contracts import (
    ArtifactContract,
    EvalConfig,
    TaskSpec,
    write_eval_config_matrix,
    write_taskbank,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_sample_taskbank(path: Path) -> dict:
    tasks = [
        TaskSpec(
            task_id="sample.swe.pytest",
            task_type="swe",
            description="Run a targeted unit-test slice for deterministic patch validation.",
            command="./eidosian_venv/bin/python -m pytest -q code_forge/tests/test_library_db.py",
            workdir=".",
            timeout_sec=1200,
            tags=("sample", "swe", "pytest"),
            contract=ArtifactContract(
                require_zero_exit=True,
                stderr_must_not_contain=("Traceback",),
            ),
        ),
        TaskSpec(
            task_id="sample.docs.linkcheck",
            task_type="docs",
            description="Validate docs index integrity by checking that core file exists and is readable.",
            command="test -f docs/DIRECTORY_INDEX_FULL.txt && wc -l docs/DIRECTORY_INDEX_FULL.txt",
            workdir=".",
            timeout_sec=300,
            tags=("sample", "docs", "integrity"),
            contract=ArtifactContract(
                require_zero_exit=True,
                stdout_must_contain=("DIRECTORY_INDEX_FULL.txt",),
            ),
        ),
    ]
    return write_taskbank(
        path=Path(path),
        tasks=tasks,
        metadata={
            "generated_at": _utc_now(),
            "description": "Sample taskbank for Code Forge eval/observability OS.",
        },
    )


def create_sample_config_matrix(path: Path) -> dict:
    configs = [
        EvalConfig(
            config_id="cortex_only",
            name="cortex_only",
            toggles={
                "CACHE_MODE": "off",
                "MEMORY_POLICY": "none",
                "LOCAL_MODELS": "off",
                "RETRIEVAL": "off",
                "VALIDATORS": "minimal",
                "PARALLEL_AGENTS": 1,
            },
            metadata={"description": "Frontier model only baseline."},
        ),
        EvalConfig(
            config_id="forge_on",
            name="forge_on",
            toggles={
                "CACHE_MODE": "swr",
                "MEMORY_POLICY": "hybrid",
                "LOCAL_MODELS": "critic",
                "RETRIEVAL": "hybrid_rerank",
                "VALIDATORS": "all",
                "PARALLEL_AGENTS": 2,
            },
            metadata={"description": "Full forge substrate enabled."},
        ),
    ]
    return write_eval_config_matrix(
        path=Path(path),
        configs=configs,
        metadata={
            "generated_at": _utc_now(),
            "description": "Sample ablation matrix for Code Forge eval harness.",
        },
    )
