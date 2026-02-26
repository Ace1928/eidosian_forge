from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Set


@dataclass
class ScribeConfig:
    forge_root: Path
    source_root: Path
    runtime_root: Path
    staging_root: Path
    final_root: Path
    rejected_root: Path
    judgments_root: Path
    index_path: Path
    state_path: Path
    status_path: Path
    host: str
    port: int
    completion_url: str
    llm_model_path: Path
    llama_server_bin: Path
    llm_server_port: int
    llm_ctx_size: int
    llm_temperature: float
    llm_n_predict: int
    include_suffixes: Set[str]
    max_file_bytes: int
    max_chars: int
    loop_sleep_s: float
    approval_threshold: float
    min_grounding_overlap: float
    enable_managed_llm: bool
    dry_run: bool
    model_lock_path: Path

    @classmethod
    def from_env(
        cls,
        forge_root: Path | None = None,
        host: str | None = None,
        port: int | None = None,
        dry_run: bool = False,
    ) -> "ScribeConfig":
        forge_root = (
            forge_root
            or Path(os.environ.get("EIDOS_FORGE_ROOT", "/data/data/com.termux/files/home/eidosian_forge")).resolve()
        )

        # Resolve default model from centralized selection, then sweep winner fallback.
        model_path_str = "models/Qwen2.5-0.5B-Instruct-Q8_0.gguf"
        selected_from_contract = False
        selection_path = forge_root / "config" / "model_selection.json"
        if selection_path.exists():
            try:
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                selected = (
                    ((selection.get("services") or {}).get("doc_forge") or {}).get("completion_model") or ""
                ).strip()
                if selected and (forge_root / selected).exists():
                    model_path_str = selected
                    selected_from_contract = True
            except Exception:
                pass

        # Load winner model from sweep only when no explicit contract selection is available.
        sweep_result = forge_root / "reports/graphrag_sweep/model_selection_latest.json"
        if not selected_from_contract and sweep_result.exists():
            try:
                data = json.loads(sweep_result.read_text(encoding="utf-8"))
                winner = data.get("winner", {}).get("model_path")
                if winner and (forge_root / winner).exists():
                    model_path_str = winner
            except Exception:
                pass

        model_path = Path(os.environ.get("EIDOS_DOC_FORGE_MODEL", model_path_str))
        if not model_path.is_absolute():
            model_path = forge_root / model_path

        runtime_root = forge_root / "doc_forge" / "runtime"

        # Port resolution would typically happen via registry, but we accept override
        default_port = 8930

        return cls(
            forge_root=forge_root,
            source_root=Path(os.environ.get("EIDOS_DOC_FORGE_SOURCE_ROOT", str(forge_root))).resolve(),
            runtime_root=runtime_root,
            staging_root=runtime_root / "staging_docs",
            final_root=runtime_root / "final_docs",
            rejected_root=runtime_root / "rejected_docs",
            judgments_root=runtime_root / "judgments",
            index_path=runtime_root / "doc_index.json",
            state_path=runtime_root / "processor_state.json",
            status_path=runtime_root / "processor_status.json",
            host=host or os.environ.get("EIDOS_DOC_FORGE_HOST", "127.0.0.1"),
            port=port if port is not None else int(os.environ.get("EIDOS_DOC_FORGE_PORT", str(default_port))),
            completion_url=os.environ.get(
                "EIDOS_DOC_FORGE_COMPLETION_URL",
                f"http://127.0.0.1:{os.environ.get('EIDOS_DOC_FORGE_LLM_PORT', '8093')}/completion",
            ),
            llm_model_path=model_path,
            llama_server_bin=(
                forge_root / os.environ.get("EIDOS_DOC_FORGE_LLAMA_SERVER_BIN", "llama.cpp/build/bin/llama-server")
            ).resolve(),
            llm_server_port=int(os.environ.get("EIDOS_DOC_FORGE_LLM_PORT", "8093")),
            llm_ctx_size=int(os.environ.get("EIDOS_DOC_FORGE_CTX_SIZE", "4096")),
            llm_temperature=float(os.environ.get("EIDOS_DOC_FORGE_TEMPERATURE", "0.2")),
            llm_n_predict=int(os.environ.get("EIDOS_DOC_FORGE_N_PREDICT", "2200")),
            include_suffixes={
                ".py",
                ".pyi",
                ".js",
                ".jsx",
                ".ts",
                ".tsx",
                ".java",
                ".kt",
                ".kts",
                ".go",
                ".rs",
                ".c",
                ".h",
                ".hpp",
                ".cc",
                ".cpp",
                ".cs",
                ".swift",
                ".rb",
                ".php",
                ".lua",
                ".sh",
                ".bash",
                ".zsh",
                ".ps1",
                ".sql",
                ".md",
                ".rst",
                ".txt",
                ".adoc",
                ".org",
                ".log",
                ".json",
                ".jsonl",
                ".yaml",
                ".yml",
                ".toml",
                ".ini",
                ".cfg",
                ".conf",
                ".xml",
                ".csv",
                ".tsv",
                ".env",
                ".html",
                ".htm",
                ".xhtml",
                ".css",
                ".scss",
                ".less",
                ".svg",
                ".pdf",
                ".docx",
            },
            max_file_bytes=int(os.environ.get("EIDOS_DOC_FORGE_MAX_FILE_BYTES", str(4 * 1024 * 1024))),
            max_chars=int(os.environ.get("EIDOS_DOC_FORGE_MAX_CHARS", "45000")),
            loop_sleep_s=float(os.environ.get("EIDOS_DOC_FORGE_LOOP_SLEEP_S", "0.2")),
            approval_threshold=float(os.environ.get("EIDOS_DOC_FORGE_APPROVAL_THRESHOLD", "0.78")),
            min_grounding_overlap=float(os.environ.get("EIDOS_DOC_FORGE_MIN_GROUNDING_OVERLAP", "0.08")),
            enable_managed_llm=os.environ.get("EIDOS_DOC_FORGE_ENABLE_MANAGED_LLM", "1") == "1",
            dry_run=dry_run,
            model_lock_path=runtime_root / "llm_server.lock",
        )
