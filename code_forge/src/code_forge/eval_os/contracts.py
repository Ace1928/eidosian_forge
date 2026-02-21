from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TASKBANK_SCHEMA_VERSION = "code_forge_taskbank_v1"
CONFIG_MATRIX_SCHEMA_VERSION = "code_forge_eval_config_matrix_v1"


def _stable_payload_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


@dataclass(frozen=True)
class ArtifactContract:
    """Artifact-level success contract for task execution."""

    require_zero_exit: bool = True
    required_paths: tuple[str, ...] = ()
    forbidden_paths: tuple[str, ...] = ()
    stdout_must_contain: tuple[str, ...] = ()
    stderr_must_not_contain: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ArtifactContract":
        payload = payload or {}
        return cls(
            require_zero_exit=bool(payload.get("require_zero_exit", True)),
            required_paths=tuple(str(v) for v in (payload.get("required_paths") or [])),
            forbidden_paths=tuple(
                str(v) for v in (payload.get("forbidden_paths") or [])
            ),
            stdout_must_contain=tuple(
                str(v) for v in (payload.get("stdout_must_contain") or [])
            ),
            stderr_must_not_contain=tuple(
                str(v) for v in (payload.get("stderr_must_not_contain") or [])
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "require_zero_exit": self.require_zero_exit,
            "required_paths": list(self.required_paths),
            "forbidden_paths": list(self.forbidden_paths),
            "stdout_must_contain": list(self.stdout_must_contain),
            "stderr_must_not_contain": list(self.stderr_must_not_contain),
        }

    def evaluate(
        self,
        *,
        workdir: Path,
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> dict[str, Any]:
        violations: list[str] = []
        if self.require_zero_exit and int(returncode) != 0:
            violations.append(f"require_zero_exit violated: returncode={returncode}")

        for rel in self.required_paths:
            if not (workdir / rel).exists():
                violations.append(f"required path missing: {rel}")
        for rel in self.forbidden_paths:
            if (workdir / rel).exists():
                violations.append(f"forbidden path exists: {rel}")
        for token in self.stdout_must_contain:
            if token not in stdout:
                violations.append(f"stdout missing token: {token}")
        for token in self.stderr_must_not_contain:
            if token in stderr:
                violations.append(f"stderr contained forbidden token: {token}")

        return {
            "pass": len(violations) == 0,
            "violations": violations,
        }


@dataclass(frozen=True)
class TaskSpec:
    """Declarative evaluation task."""

    task_id: str
    task_type: str
    description: str
    command: str
    workdir: str = "."
    timeout_sec: int = 1200
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    contract: ArtifactContract = field(default_factory=ArtifactContract)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskSpec":
        task_id = str(payload.get("task_id") or "").strip()
        if not task_id:
            raise ValueError("task.task_id is required")
        command = str(payload.get("command") or "").strip()
        if not command:
            raise ValueError(f"task {task_id}: command is required")
        task_type = str(payload.get("task_type") or "hybrid").strip().lower()
        if task_type not in {"swe", "docs", "hybrid"}:
            raise ValueError(f"task {task_id}: invalid task_type={task_type!r}")
        timeout_sec = int(payload.get("timeout_sec") or 1200)
        if timeout_sec <= 0:
            raise ValueError(f"task {task_id}: timeout_sec must be > 0")
        return cls(
            task_id=task_id,
            task_type=task_type,
            description=str(payload.get("description") or "").strip(),
            command=command,
            workdir=str(payload.get("workdir") or "."),
            timeout_sec=timeout_sec,
            tags=tuple(str(v) for v in (payload.get("tags") or [])),
            metadata=dict(payload.get("metadata") or {}),
            contract=ArtifactContract.from_dict(payload.get("contract")),
        )

    @property
    def task_hash(self) -> str:
        return _stable_payload_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "description": self.description,
            "command": self.command,
            "workdir": self.workdir,
            "timeout_sec": self.timeout_sec,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "contract": self.contract.to_dict(),
        }


@dataclass(frozen=True)
class EvalConfig:
    """Single configuration in the ablation/config matrix."""

    config_id: str
    name: str
    toggles: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvalConfig":
        name = str(payload.get("name") or "").strip()
        if not name:
            raise ValueError("config.name is required")
        toggles = dict(payload.get("toggles") or {})
        if not toggles:
            raise ValueError(f"config {name}: toggles must not be empty")
        cfg_id = str(payload.get("config_id") or "").strip()
        if not cfg_id:
            cfg_id = _stable_payload_hash({"name": name, "toggles": toggles})
        return cls(
            config_id=cfg_id,
            name=name,
            toggles=toggles,
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "name": self.name,
            "toggles": dict(self.toggles),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EvalConfigMatrix:
    schema_version: str
    configs: tuple[EvalConfig, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvalConfigMatrix":
        schema_version = str(payload.get("schema_version") or "")
        if schema_version != CONFIG_MATRIX_SCHEMA_VERSION:
            raise ValueError(
                f"invalid config matrix schema_version={schema_version!r}; "
                f"expected {CONFIG_MATRIX_SCHEMA_VERSION!r}"
            )
        configs = tuple(
            EvalConfig.from_dict(item) for item in (payload.get("configs") or [])
        )
        if not configs:
            raise ValueError("config matrix must include at least one config")
        seen: set[str] = set()
        for cfg in configs:
            if cfg.config_id in seen:
                raise ValueError(f"duplicate config_id={cfg.config_id}")
            seen.add(cfg.config_id)
        return cls(
            schema_version=schema_version,
            configs=configs,
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "configs": [cfg.to_dict() for cfg in self.configs],
            "metadata": dict(self.metadata),
        }


def load_taskbank(path: Path) -> tuple[str, list[TaskSpec], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema_version = str(payload.get("schema_version") or "")
    if schema_version != TASKBANK_SCHEMA_VERSION:
        raise ValueError(
            f"invalid taskbank schema_version={schema_version!r}; expected {TASKBANK_SCHEMA_VERSION!r}"
        )
    tasks = [TaskSpec.from_dict(item) for item in (payload.get("tasks") or [])]
    if not tasks:
        raise ValueError("taskbank contains no tasks")
    seen: set[str] = set()
    for task in tasks:
        if task.task_id in seen:
            raise ValueError(f"duplicate task_id={task.task_id}")
        seen.add(task.task_id)
    metadata = dict(payload.get("metadata") or {})
    return schema_version, tasks, metadata


def write_taskbank(
    path: Path, tasks: list[TaskSpec], metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    if not tasks:
        raise ValueError("cannot write empty taskbank")
    payload = {
        "schema_version": TASKBANK_SCHEMA_VERSION,
        "metadata": dict(metadata or {}),
        "tasks": [task.to_dict() for task in tasks],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_eval_config_matrix(path: Path) -> EvalConfigMatrix:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalConfigMatrix.from_dict(payload)


def write_eval_config_matrix(
    path: Path,
    configs: list[EvalConfig],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not configs:
        raise ValueError("cannot write empty config matrix")
    payload = {
        "schema_version": CONFIG_MATRIX_SCHEMA_VERSION,
        "metadata": dict(metadata or {}),
        "configs": [cfg.to_dict() for cfg in configs],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
