from __future__ import annotations

import json
from pathlib import Path

from code_forge.eval_os.contracts import ArtifactContract, EvalConfig, TaskSpec, write_eval_config_matrix, write_taskbank
from code_forge.eval_os.runner import EvalRunOptions, run_eval_suite
from code_forge.eval_os.tracing import export_trace_jsonl_to_otlp


def test_export_trace_jsonl_to_otlp_success(tmp_path: Path, monkeypatch) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": "2026-02-22T10:00:00Z",
                        "event_type": "command.start",
                        "trace_id": "0123456789abcdef0123456789abcdef",
                        "run_id": "run-1",
                        "task_id": "task-1",
                        "config_id": "cfg-1",
                        "span_id": "0123456789abcdef",
                        "parent_span_id": None,
                        "name": "command.execute",
                        "status": "start",
                        "attributes": {"command": "echo ok"},
                    }
                ),
                json.dumps(
                    {
                        "ts": "2026-02-22T10:00:01Z",
                        "event_type": "command.end",
                        "trace_id": "0123456789abcdef0123456789abcdef",
                        "run_id": "run-1",
                        "task_id": "task-1",
                        "config_id": "cfg-1",
                        "span_id": "0123456789abcdef",
                        "parent_span_id": None,
                        "name": "command.execute",
                        "status": "ok",
                        "attributes": {"duration_ms": 1000.0},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    captured: dict[str, str] = {}

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b"ok"

        def getcode(self) -> int:
            return 200

    def _fake_urlopen(req, timeout=10):
        captured["url"] = req.full_url
        captured["body"] = (req.data or b"").decode("utf-8")
        return _Resp()

    monkeypatch.setattr("code_forge.eval_os.tracing.urlrequest.urlopen", _fake_urlopen)

    result = export_trace_jsonl_to_otlp(
        trace_path=trace_path,
        endpoint="http://127.0.0.1:4318",
        service_name="code_forge_eval_test",
        headers={"X-Test": "1"},
        timeout_sec=5,
    )
    assert result["ok"] is True
    assert result["status_code"] == 200
    assert result["spans"] >= 2
    assert captured["url"].endswith("/v1/traces")
    payload = json.loads(captured["body"])
    spans = (((payload.get("resourceSpans") or [{}])[0].get("scopeSpans") or [{}])[0].get("spans")) or []
    assert len(spans) >= 2


def test_run_eval_suite_otlp_wiring(tmp_path: Path, monkeypatch) -> None:
    taskbank = tmp_path / "taskbank.json"
    matrix = tmp_path / "config_matrix.json"
    write_taskbank(
        taskbank,
        tasks=[
            TaskSpec(
                task_id="eval.otlp.ok",
                task_type="swe",
                description="simple success command",
                command="printf 'OTLP_OK\\n'",
                contract=ArtifactContract(
                    require_zero_exit=True,
                    stdout_must_contain=("OTLP_OK",),
                ),
            )
        ],
        metadata={"suite": "otlp"},
    )
    write_eval_config_matrix(
        matrix,
        configs=[EvalConfig(config_id="cfg_otlp", name="cfg_otlp", toggles={"CACHE_MODE": "off"})],
        metadata={"suite": "otlp"},
    )

    def _fake_export(**kwargs):
        return {
            "ok": True,
            "endpoint": kwargs.get("endpoint"),
            "status_code": 200,
            "events": 2,
            "spans": 2,
        }

    monkeypatch.setattr("code_forge.eval_os.runner.export_trace_jsonl_to_otlp", _fake_export)

    payload = run_eval_suite(
        EvalRunOptions(
            taskbank_path=taskbank,
            config_matrix_path=matrix,
            output_dir=tmp_path / "out",
            repo_root=tmp_path,
            repeats=1,
            replay_mode="off",
            max_parallel=1,
            otlp_endpoint="http://127.0.0.1:4318",
            otlp_service_name="code_forge_eval_test",
            otlp_timeout_sec=5,
            otlp_headers={"x-token": "abc"},
        )
    )

    run_stats = payload.get("run_stats") or {}
    otlp = run_stats.get("otlp") or {}
    assert otlp.get("enabled") is True
    assert otlp.get("attempted") == 1
    assert otlp.get("ok") == 1
    assert otlp.get("failed") == 0
