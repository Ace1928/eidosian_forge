from __future__ import annotations

import argparse
import os
import subprocess
import threading
import time
import hashlib
from contextlib import asynccontextmanager
from collections import deque
from pathlib import Path
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse

from .config import ScribeConfig
from .extract import DocumentExtractor, extract_terms
from .generate import DocGenerator
from .judge import FederatedJudge
from .state import ProcessorState, atomic_write_json, now_iso, read_json

def _atlas_dashboard_url() -> str:
    explicit = os.environ.get("EIDOS_ATLAS_URL", "").strip()
    if explicit:
        return explicit
    port = os.environ.get("EIDOS_ATLAS_PORT", "8936")
    host = os.environ.get("EIDOS_ATLAS_HOST", "127.0.0.1")
    return f"http://{host}:{port}/"

class ManagedModelServer:
    def __init__(self, cfg: ScribeConfig) -> None:
        self.cfg = cfg
        self.proc: subprocess.Popen[Any] | None = None
        self.log_handle: Any = None

    def _http_ready(self, timeout_s: float = 1.5) -> bool:
        try:
            r = requests.get(self.cfg.completion_url.replace("/completion", "/health"), timeout=timeout_s)
            return r.status_code < 500
        except Exception:
            return False

    def is_ready(self, timeout_s: float = 1.0) -> bool:
        return self._http_ready(timeout_s=timeout_s)

    def start(self) -> None:
        if self._http_ready():
            return
        if not self.cfg.enable_managed_llm:
            raise RuntimeError("LLM completion endpoint unavailable and managed startup disabled")
        if not self.cfg.llama_server_bin.exists():
            raise RuntimeError(f"Missing llama-server binary: {self.cfg.llama_server_bin}")
        if not self.cfg.llm_model_path.exists():
            raise RuntimeError(f"Missing doc processor model: {self.cfg.llm_model_path}")

        self.cfg.runtime_root.mkdir(parents=True, exist_ok=True)
        log_path = self.cfg.runtime_root / "llama_server.log"
        self.log_handle = log_path.open("a")
        
        # Determine parallel setting: if low ram/cpu, stick to 1, else 2
        parallel = 1
        
        cmd = [
            str(self.cfg.llama_server_bin),
            "-m", str(self.cfg.llm_model_path),
            "--port", str(self.cfg.llm_server_port),
            "--ctx-size", str(self.cfg.llm_ctx_size),
            "--temp", str(self.cfg.llm_temperature),
            "--n-gpu-layers", os.environ.get("EIDOS_LLAMA_GPU_LAYERS", "0"),
            "--parallel", str(parallel),
        ]

        env = os.environ.copy()
        bin_dir = str(self.cfg.llama_server_bin.resolve().parent)
        env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
        ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{bin_dir}:{ld}" if ld else bin_dir

        self.proc = subprocess.Popen(cmd, stdout=self.log_handle, stderr=subprocess.STDOUT, env=env)
        start = time.time()
        while time.time() - start < 90:
            if self._http_ready(timeout_s=2.0):
                return
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode}")
            time.sleep(0.5)
        raise TimeoutError("Timed out waiting for managed llama-server")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None

class DocProcessor:
    def __init__(self, cfg: ScribeConfig) -> None:
        self.cfg = cfg
        self.extractor = DocumentExtractor(cfg)
        self.generator = DocGenerator(cfg)
        self.judges = FederatedJudge(cfg)
        self.model_server = ManagedModelServer(cfg)
        self.state = ProcessorState(cfg.state_path, cfg.status_path, cfg.index_path)
        self.state.update("model", str(cfg.llm_model_path))
        self.state.update("completion_url", cfg.completion_url)
        
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.recent_docs: deque[dict[str, Any]] = deque(maxlen=80)

        # Exclusion config
        self.excluded_prefixes = [
            (cfg.forge_root / ".git").resolve(),
            (cfg.forge_root / "eidosian_venv").resolve(),
            (cfg.forge_root / "llama.cpp").resolve(),
            (cfg.forge_root / "models").resolve(),
            (cfg.forge_root / "doc_forge" / "staging").resolve(),
            (cfg.forge_root / "doc_forge" / "final_docs").resolve(),
            cfg.runtime_root.resolve(),
        ]
        self.excluded_segments = {
            ".git", "__pycache__", ".pytest_cache", "node_modules", "archive_forge", "Backups", 
            "runtime", "staging", "final_docs", "eidos_mcp_backup"
        }

        for path in [cfg.runtime_root, cfg.staging_root, cfg.final_root, cfg.rejected_root, cfg.judgments_root]:
            path.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, name="doc-processor", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=10)
        self.model_server.stop()

    def _is_excluded(self, path: Path) -> bool:
        resolved = path.resolve()
        for prefix in self.excluded_prefixes:
            try:
                resolved.relative_to(prefix)
                return True
            except Exception:
                pass
        for part in resolved.parts:
            if part in self.excluded_segments:
                return True
        return False

    def _should_include(self, path: Path) -> bool:
        if not path.is_file():
            return False
        if self._is_excluded(path):
            return False
        suffix = path.suffix.lower()
        if suffix not in self.cfg.include_suffixes and suffix != "":
            return False
        try:
            size = path.stat().st_size
        except OSError:
            return False
        if size <= 0:
            return False
        if size > self.cfg.max_file_bytes and suffix not in {".pdf", ".docx"}:
            return False
        return True

    def _scan_candidates(self) -> list[Path]:
        out: list[Path] = []
        for cur, dirs, files in os.walk(self.cfg.source_root):
            cur_path = Path(cur)
            dirs[:] = [d for d in dirs if not self._is_excluded(cur_path / d)]
            for filename in files:
                path = cur_path / filename
                if self._should_include(path):
                    out.append(path)
        out.sort()
        return out

    def _run_loop(self) -> None:
        self.state.update("status", "starting")
        self.state.persist()

        while not self.stop_event.is_set():
            if not self.cfg.dry_run and not self.model_server.is_ready():
                try:
                    self.model_server.start()
                    self.state.update("status", "running")
                    self.state.update("last_error", "")
                except Exception as exc:
                    self.state.update("status", "waiting_for_llm")
                    self.state.update("last_error", f"{type(exc).__name__}: {exc}")
                    self.state.persist()
                    time.sleep(max(2.0, self.cfg.loop_sleep_s))
                    continue
                self.state.persist()
            elif self.cfg.dry_run:
                 if self.state.get("status") == "starting":
                        self.state.update("status", "running")
                        self.state.persist()

            candidates = self._scan_candidates()
            self.state.update("total_discovered", len(candidates))

            pending = []
            for path in candidates:
                try:
                    rel = path.resolve().relative_to(self.cfg.source_root.resolve())
                    rel_key = str(rel)
                    raw = path.read_bytes()
                    digest = hashlib.sha256(raw).hexdigest()
                    file_state = self.state.get("files", {}).get(rel_key)
                    if file_state and file_state.get("sha256") == digest and file_state.get("status") == "approved":
                        continue
                    pending.append((path, rel, digest))
                except Exception:
                    continue

            if not pending:
                self.state.update("status", "idle")
                self.state.persist()
                time.sleep(max(2.0, self.cfg.loop_sleep_s))
                if self.state.get("status") == "idle":
                    self.state.update("status", "running")
                continue

            for path, rel, digest in pending:
                if self.stop_event.is_set():
                    break
                started = time.perf_counter()
                rel_key = str(rel)
                
                # Paths
                stage_path = self.cfg.staging_root / rel.with_suffix(rel.suffix + ".md")
                final_path = self.cfg.final_root / rel.with_suffix(rel.suffix + ".md")
                rejected_path = self.cfg.rejected_root / rel.with_suffix(rel.suffix + ".md")
                judgment_path = self.cfg.judgments_root / rel.with_suffix(rel.suffix + ".json")
                
                for p in (stage_path, final_path, rejected_path, judgment_path):
                    p.parent.mkdir(parents=True, exist_ok=True)

                doc_type = rel.suffix.lower().lstrip(".") or "text"
                self.state.data["doc_type_frequency"][doc_type] = self.state.data["doc_type_frequency"].get(doc_type, 0) + 1

                try:
                    source_text, metadata = self.extractor.extract(path)
                    metadata["source_size_bytes"] = path.stat().st_size
                    metadata["doc_type"] = metadata.get("doc_type") or doc_type
                    
                    markdown = (
                        self.generator.generate(rel_key, source_text, metadata)
                        if not self.cfg.dry_run
                        else "\n".join(["# Placeholder", "## Content"])
                    )

                    stage_path.write_text(markdown, encoding="utf-8")
                    self.state.update("last_staged", rel_key)

                    scorecard = self.judges.evaluate(
                        markdown=markdown, source_text=source_text, rel_path=rel_key, metadata=metadata
                    )
                    atomic_write_json(judgment_path, scorecard)

                    quality = float(scorecard.get("aggregate_score", 0.0))
                    self.state.update_running_average("average_quality_score", quality, "quality_samples")

                    tags = extract_terms(markdown, limit=8)
                    themes = extract_terms(source_text, limit=8)
                    
                    with self.state.lock:
                        for t in tags:
                            self.state.data["tag_frequency"][t] = self.state.data["tag_frequency"].get(t, 0) + 1
                        for t in themes:
                            self.state.data["theme_frequency"][t] = self.state.data["theme_frequency"].get(t, 0) + 1

                    approved = bool(scorecard.get("approved", False))
                    status = "approved" if approved else "rejected"
                    
                    if approved:
                        final_path.write_text(markdown, encoding="utf-8")
                        with self.state.lock:
                            self.state.data["approved"] += 1
                            self.state.data["last_approved"] = rel_key
                        
                        index_entry = {
                            "source": rel_key,
                            "document": str(final_path.relative_to(self.cfg.runtime_root)),
                            "score": quality,
                            "updated_at": now_iso(),
                            "tags": tags,
                            "doc_type": metadata.get("doc_type", "unknown"),
                        }
                        with self.state.lock:
                            current_index = [e for e in self.state.data.get("index", []) if e.get("source") != rel_key]
                            current_index.append(index_entry)
                            current_index.sort(key=lambda x: x.get("source", ""))
                            self.state.data["index"] = current_index
                    else:
                        rejected_path.write_text(markdown, encoding="utf-8")
                        with self.state.lock:
                            self.state.data["rejected"] += 1

                    elapsed = round(time.perf_counter() - started, 4)
                    self.state.update_running_average("average_seconds_per_document", elapsed, "duration_samples")

                    with self.state.lock:
                        self.state.data["processed"] += 1
                        self.state.data["files"][rel_key] = {
                            "sha256": digest,
                            "status": status,
                            "score": quality,
                            "doc_type": metadata.get("doc_type", "unknown"),
                            "updated_at": now_iso(),
                            "duration_seconds": elapsed,
                            "tags": tags,
                        }
                        history_entry = {
                            "source": rel_key,
                            "status": status,
                            "score": quality,
                            "duration_seconds": elapsed,
                            "updated_at": now_iso(),
                        }
                        self.state.data["history"] = (self.state.data.get("history", []) + [history_entry])[-500:]

                except Exception as exc:
                    elapsed = round(time.perf_counter() - started, 4)
                    with self.state.lock:
                        self.state.data["processed"] += 1
                        self.state.data["errors"] += 1
                        self.state.data["files"][rel_key] = {
                            "sha256": digest,
                            "status": "error",
                            "error": str(exc),
                            "updated_at": now_iso(),
                            "duration_seconds": elapsed,
                        }
                        
                self.state.persist()
                time.sleep(self.cfg.loop_sleep_s)

        self.state.update("status", "stopped")
        self.state.persist()

def create_app(processor: DocProcessor) -> FastAPI:
    @asynccontextmanager
    async def _lifespan(_: FastAPI):
        processor.start()
        try:
            yield
        finally:
            processor.stop()

    app = FastAPI(title="Eidosian Doc Processor", version="3.0.0", lifespan=_lifespan)

    @app.get("/health")
    def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "service": "doc_processor", "time": now_iso()})

    @app.get("/api/status")
    def api_status() -> JSONResponse:
        return JSONResponse(read_json(processor.cfg.status_path, {}))

    @app.get("/")
    def dashboard() -> RedirectResponse:
        return RedirectResponse(url=_atlas_dashboard_url(), status_code=307)

    return app

def main() -> int:
    parser = argparse.ArgumentParser(description="Run Eidosian Doc Processor v3 service")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = ScribeConfig.from_env(host=args.host, port=args.port, dry_run=args.dry_run)
    processor = DocProcessor(cfg)
    app = create_app(processor)

    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
