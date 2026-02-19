from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from code_forge.analyzer.generic_analyzer import GenericCodeAnalyzer
from code_forge.analyzer.python_analyzer import CodeAnalyzer
from code_forge.library.db import CodeLibraryDB, CodeUnit
from code_forge.library.similarity import build_fingerprint, structural_hash

FORGE_ROOT = Path(os.environ.get("EIDOS_FORGE_DIR", str(Path(__file__).resolve().parents[4]))).resolve()
DEFAULT_RUNS_DIR = FORGE_ROOT / "data" / "code_forge" / "ingestion_runs"
ANALYSIS_VERSION = 3
DEFAULT_EXTENSIONS = GenericCodeAnalyzer.supported_extensions()


@dataclass
class IngestionStats:
    run_id: str
    root_path: str
    mode: str
    files_processed: int
    units_created: int
    errors: int
    files_total: int = 0
    loc_total: int = 0
    bytes_total: int = 0
    files_skipped: int = 0
    interrupted: bool = False
    elapsed_seconds: float = 0.0
    eta_seconds: Optional[float] = None


class IngestionRunner:
    """Non-destructive ingestion runner with multi-language support."""

    def __init__(
        self,
        db: CodeLibraryDB,
        runs_dir: Path = DEFAULT_RUNS_DIR,
    ) -> None:
        self.db = db
        self.python_analyzer = CodeAnalyzer()
        self.generic_analyzer = GenericCodeAnalyzer()
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._external_unit_cache: Dict[str, str] = {}

    @staticmethod
    def _module_qualified_name(rel_path: str) -> str:
        without_ext = rel_path.replace("\\", "/")
        if "." in without_ext:
            without_ext = without_ext.rsplit(".", 1)[0]
        return without_ext.replace("/", ".")

    @staticmethod
    def _normalize_qualified(module_qn: str, value: Optional[str], fallback_name: str) -> str:
        candidate = (value or fallback_name or "").strip()
        if not candidate:
            return module_qn
        if candidate == module_qn or candidate.startswith(module_qn + "."):
            return candidate
        if candidate.startswith("."):
            candidate = candidate.lstrip(".")
        return f"{module_qn}.{candidate}" if module_qn else candidate

    @staticmethod
    def _normalize_parent(module_qn: str, value: Optional[str]) -> Optional[str]:
        if value is None:
            return module_qn
        candidate = str(value).strip()
        if not candidate:
            return module_qn
        if candidate == module_qn or candidate.startswith(module_qn + "."):
            return candidate
        if candidate.startswith("."):
            candidate = candidate.lstrip(".")
        return f"{module_qn}.{candidate}" if module_qn else candidate

    def _write_manifest(self, stats: IngestionStats, files: List[str]) -> Path:
        path = self.runs_dir / f"{stats.run_id}.json"
        payload = asdict(stats)
        payload["files"] = files
        path.write_text(json.dumps(payload, indent=2))
        return path

    def _iter_source_files(
        self,
        root_path: Path,
        extensions: Iterable[str],
        exclude_patterns: List[str],
    ) -> Iterable[Path]:
        extensions = {e.lower() for e in extensions}
        for file_path in root_path.rglob("*"):
            if not file_path.is_file():
                continue
            if self._is_excluded(file_path, exclude_patterns):
                continue
            if file_path.suffix.lower() in extensions:
                yield file_path

    @staticmethod
    def _is_excluded(file_path: Path, exclude_patterns: Iterable[str]) -> bool:
        full = str(file_path)
        parts = set(file_path.parts)
        for pattern in exclude_patterns:
            token = str(pattern or "").strip()
            if not token:
                continue
            if "/" in token:
                if token in full:
                    return True
                continue
            if token in parts:
                return True
        return False

    def _scan_files(
        self,
        root_path: Path,
        extensions: Iterable[str],
        exclude_patterns: List[str],
        max_files: Optional[int] = None,
        estimate_loc: bool = True,
        sample_size: int = 200,
    ) -> Tuple[List[Tuple[Path, int, int]], int, int]:
        files: List[Tuple[Path, int, int]] = []
        total_bytes = 0
        total_loc = 0
        sample_lines = 0
        sample_bytes = 0
        seen = 0

        for path in self._iter_source_files(root_path, extensions, exclude_patterns):
            try:
                file_bytes = path.stat().st_size
                if estimate_loc:
                    file_loc = 0
                else:
                    file_loc = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore"))
            except OSError:
                continue

            if estimate_loc and seen < sample_size:
                try:
                    lines = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore"))
                    sample_lines += lines
                    sample_bytes += file_bytes
                    seen += 1
                except OSError:
                    pass

            files.append((path, file_loc, file_bytes))
            total_bytes += file_bytes
            total_loc += file_loc
            if max_files is not None and len(files) >= max_files:
                break

        if estimate_loc:
            avg_bytes_per_line = (sample_bytes / sample_lines) if sample_lines > 0 else 80.0
            total_loc = 0
            updated_files: List[Tuple[Path, int, int]] = []
            for path, _loc, file_bytes in files:
                est_loc = max(1, int(file_bytes / avg_bytes_per_line))
                updated_files.append((path, est_loc, file_bytes))
                total_loc += est_loc
            files = updated_files

        return files, total_loc, total_bytes

    def _analyze_file_by_language(self, file_path: Path, language: str) -> Dict[str, object]:
        if language == "python":
            return self.python_analyzer.analyze_file(file_path)
        return self.generic_analyzer.analyze_file(file_path)

    def ingest_file(self, file_path: Path, root_path: Path, run_id: str) -> int:
        source_text = file_path.read_text(encoding="utf-8", errors="ignore")
        file_hash = self.db.add_text(source_text)
        rel_path = str(file_path.relative_to(root_path))
        abs_path = str(file_path)
        if not self.db.should_process_file(abs_path, file_hash, ANALYSIS_VERSION):
            return 0

        language = GenericCodeAnalyzer.detect_language(file_path)
        analysis = self._analyze_file_by_language(file_path, language)
        if "error" in analysis:
            return 0

        created = 0
        module_qn = self._module_qualified_name(rel_path)

        module_meta = analysis.get("module", {}) if isinstance(analysis, dict) else {}
        module_hash = file_hash
        norm_hash, simhash, token_count = build_fingerprint(source_text)
        module_struct_hash = structural_hash(source_text)

        module_unit = CodeUnit(
            unit_type="module",
            name=file_path.stem,
            qualified_name=module_qn,
            file_path=rel_path,
            language=language,
            line_start=module_meta.get("line_start") if isinstance(module_meta, dict) else None,
            line_end=module_meta.get("line_end") if isinstance(module_meta, dict) else None,
            col_start=module_meta.get("col_start") if isinstance(module_meta, dict) else None,
            col_end=module_meta.get("col_end") if isinstance(module_meta, dict) else None,
            content_hash=module_hash,
            run_id=run_id,
            normalized_hash=norm_hash,
            structural_hash=module_struct_hash,
            simhash64=f"{simhash:016x}",
            token_count=token_count,
            semantic_text=source_text[:4000],
        )
        module_id = self.db.add_unit(module_unit)
        created += 1

        nodes = list(analysis.get("nodes", []) if isinstance(analysis, dict) else [])
        nodes.sort(key=lambda n: (n.get("line_start") or 0, n.get("line_end") or 0))

        id_by_qualified: Dict[str, str] = {
            module_qn: module_id,
            file_path.stem: module_id,
        }
        simple_name_to_qualified: Dict[str, str] = {}

        for node in nodes:
            node_source = str(node.get("source") or "")
            node_hash = self.db.add_text(node_source) if node_source else None
            unit_type = str(node.get("unit_type") or "node")
            name = str(node.get("name") or unit_type)

            qualified = self._normalize_qualified(module_qn, node.get("qualified_name"), name)
            parent_qn = self._normalize_parent(module_qn, node.get("parent_qualified_name"))
            parent_id = id_by_qualified.get(parent_qn, module_id)

            node_norm_hash = None
            node_struct_hash = None
            node_simhash = None
            node_token_count = None
            if node_source:
                node_norm_hash, raw_simhash, node_token_count = build_fingerprint(node_source)
                node_struct_hash = structural_hash(node_source)
                node_simhash = f"{raw_simhash:016x}"

            unit = CodeUnit(
                unit_type=unit_type,
                name=name,
                qualified_name=qualified,
                file_path=rel_path,
                language=language,
                line_start=node.get("line_start"),
                line_end=node.get("line_end"),
                col_start=node.get("col_start"),
                col_end=node.get("col_end"),
                content_hash=node_hash,
                parent_id=parent_id,
                run_id=run_id,
                complexity=node.get("complexity"),
                normalized_hash=node_norm_hash,
                structural_hash=node_struct_hash,
                simhash64=node_simhash,
                token_count=node_token_count,
                semantic_text=node_source[:2000] if node_source else None,
            )
            unit_id = self.db.add_unit(unit)
            self.db.add_relationship(parent_id, unit_id, "contains")
            id_by_qualified[qualified] = unit_id
            if name and name not in simple_name_to_qualified:
                simple_name_to_qualified[name] = qualified
            created += 1

        self._ingest_edges(
            analysis=analysis,
            module_qn=module_qn,
            module_id=module_id,
            id_by_qualified=id_by_qualified,
            simple_name_to_qualified=simple_name_to_qualified,
            run_id=run_id,
        )

        self.db.update_file_record(abs_path, file_hash, ANALYSIS_VERSION)
        return created

    @staticmethod
    def _normalize_edge_qn(module_qn: str, raw_qn: Optional[str]) -> str:
        qn = (raw_qn or "").strip()
        if not qn or qn == "__module__":
            return module_qn
        if qn == module_qn or qn.startswith(module_qn + "."):
            return qn
        if qn.startswith("."):
            qn = qn.lstrip(".")
        return f"{module_qn}.{qn}" if module_qn else qn

    def _get_or_create_external_unit(self, target: str, run_id: str) -> str:
        key = target.strip()
        if key in self._external_unit_cache:
            return self._external_unit_cache[key]

        ext_hash, ext_simhash, ext_token_count = build_fingerprint(key)
        ext_struct_hash = structural_hash(key)
        ext_content_hash = self.db.add_text(key)
        ext_unit = CodeUnit(
            unit_type="external_symbol",
            name=key.split(".")[-1] if "." in key else key,
            qualified_name=key,
            file_path="__external__/symbols",
            language="external",
            content_hash=ext_content_hash,
            run_id=run_id,
            normalized_hash=ext_hash,
            structural_hash=ext_struct_hash,
            simhash64=f"{ext_simhash:016x}",
            token_count=ext_token_count,
            semantic_text=key,
        )
        unit_id = self.db.add_unit(ext_unit)
        self._external_unit_cache[key] = unit_id
        return unit_id

    def _resolve_edge_target(
        self,
        target: str,
        module_qn: str,
        id_by_qualified: Dict[str, str],
        simple_name_to_qualified: Dict[str, str],
        run_id: str,
    ) -> str:
        candidate = target.strip()
        if not candidate:
            return ""

        if candidate in simple_name_to_qualified:
            local_qn = simple_name_to_qualified[candidate]
            unit_id = id_by_qualified.get(local_qn)
            if unit_id:
                return unit_id

        qualified_candidate = self._normalize_qualified(module_qn, candidate, candidate)
        if qualified_candidate in id_by_qualified:
            return id_by_qualified[qualified_candidate]

        if candidate in id_by_qualified:
            return id_by_qualified[candidate]

        # Try best-effort module-local normalization for dotted names.
        if "." in candidate and not candidate.startswith(module_qn + "."):
            tail = candidate.split(".")[-1]
            if tail in simple_name_to_qualified:
                local_qn = simple_name_to_qualified[tail]
                unit_id = id_by_qualified.get(local_qn)
                if unit_id:
                    return unit_id

        return self._get_or_create_external_unit(candidate, run_id)

    def _ingest_edges(
        self,
        analysis: Dict[str, object],
        module_qn: str,
        module_id: str,
        id_by_qualified: Dict[str, str],
        simple_name_to_qualified: Dict[str, str],
        run_id: str,
    ) -> None:
        edges = analysis.get("edges", []) if isinstance(analysis, dict) else []
        if not isinstance(edges, list):
            return

        # Bound edge ingestion per file to avoid pathological growth on noisy parsers.
        max_edges = 1200
        ingested = 0
        for edge in edges:
            if ingested >= max_edges:
                break
            if not isinstance(edge, dict):
                continue
            rel_type = str(edge.get("rel_type") or "").strip()
            if rel_type not in {"imports", "calls", "uses"}:
                continue

            source_qn = self._normalize_edge_qn(module_qn, edge.get("source_qualified_name"))
            source_id = id_by_qualified.get(source_qn, module_id)

            target = str(edge.get("target") or "").strip()
            if not target:
                continue
            target_id = self._resolve_edge_target(
                target=target,
                module_qn=module_qn,
                id_by_qualified=id_by_qualified,
                simple_name_to_qualified=simple_name_to_qualified,
                run_id=run_id,
            )
            if not target_id:
                continue

            self.db.add_relationship(source_id, target_id, rel_type)
            ingested += 1

    def ingest_path(
        self,
        root_path: Path,
        mode: str = "analysis",
        exclude_patterns: Optional[List[str]] = None,
        extensions: Optional[Iterable[str]] = None,
        max_files: Optional[int] = None,
        progress_every: int = 50,
        run_id: Optional[str] = None,
    ) -> IngestionStats:
        root_path = Path(root_path).resolve()
        exclude_patterns = exclude_patterns or [
            ".git",
            "__pycache__",
            ".venv",
            "venv",
            "node_modules",
            "dist",
            "build",
            ".mypy_cache",
            ".pytest_cache",
            "data/code_forge/digester",
            "data/code_forge/graphrag_input",
            "doc_forge/final_docs",
        ]
        extensions = set(ext.lower() for ext in (extensions or DEFAULT_EXTENSIONS))

        run_id = run_id or uuid.uuid4().hex[:16]
        self.db.create_run(str(root_path), mode, run_id=run_id, config={"extensions": sorted(extensions)})
        files: List[str] = []
        files_processed = 0
        units_created = 0
        errors = 0
        files_skipped = 0

        scan_files, total_loc, total_bytes = self._scan_files(
            root_path,
            extensions,
            exclude_patterns,
            max_files=max_files,
        )

        total_files = len(scan_files)
        processed_loc = 0
        processed_bytes = 0
        interrupted = False

        start_time = time.monotonic()
        initial_stats = IngestionStats(
            run_id=run_id,
            root_path=str(root_path),
            mode=mode,
            files_processed=files_processed,
            units_created=units_created,
            errors=errors,
            files_total=total_files,
            loc_total=total_loc,
            bytes_total=total_bytes,
            files_skipped=files_skipped,
            interrupted=False,
            elapsed_seconds=0.0,
            eta_seconds=None,
        )
        self._write_manifest(initial_stats, files)

        try:
            for idx, (file_path, file_loc, file_bytes) in enumerate(scan_files, start=1):
                try:
                    created = self.ingest_file(file_path, root_path, run_id)
                    if created > 0:
                        files_processed += 1
                        units_created += created
                        files.append(str(file_path))
                    else:
                        files_skipped += 1
                except Exception:
                    errors += 1

                processed_loc += file_loc
                processed_bytes += file_bytes

                if progress_every and idx % progress_every == 0:
                    elapsed = time.monotonic() - start_time
                    rate_loc = processed_loc / elapsed if elapsed > 0 else 0.0
                    remaining_loc = max(total_loc - processed_loc, 0)
                    eta = remaining_loc / rate_loc if rate_loc > 0 else None
                    self._write_manifest(
                        IngestionStats(
                            run_id=run_id,
                            root_path=str(root_path),
                            mode=mode,
                            files_processed=files_processed,
                            units_created=units_created,
                            errors=errors,
                            files_total=total_files,
                            loc_total=total_loc,
                            bytes_total=total_bytes,
                            files_skipped=files_skipped,
                            interrupted=False,
                            elapsed_seconds=elapsed,
                            eta_seconds=eta,
                        ),
                        files,
                    )
                    eta_str = f"{int(eta)}s" if eta is not None else "unknown"
                    print(
                        f"Progress: {idx}/{total_files} files, "
                        f"{processed_loc}/{total_loc} LOC, "
                        f"{processed_bytes}/{total_bytes} bytes, "
                        f"eta={eta_str}"
                    )
        except KeyboardInterrupt:
            interrupted = True

        elapsed = time.monotonic() - start_time
        rate_loc = processed_loc / elapsed if elapsed > 0 else 0.0
        remaining_loc = max(total_loc - processed_loc, 0)
        eta = remaining_loc / rate_loc if rate_loc > 0 else None
        stats = IngestionStats(
            run_id=run_id,
            root_path=str(root_path),
            mode=mode,
            files_processed=files_processed,
            units_created=units_created,
            errors=errors,
            files_total=total_files,
            loc_total=total_loc,
            bytes_total=total_bytes,
            files_skipped=files_skipped,
            interrupted=interrupted,
            elapsed_seconds=elapsed,
            eta_seconds=eta,
        )
        self._write_manifest(stats, files)
        return stats
