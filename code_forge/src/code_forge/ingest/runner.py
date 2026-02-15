from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import uuid
import time

from code_forge.analyzer.python_analyzer import CodeAnalyzer
from code_forge.library.db import CodeLibraryDB, CodeUnit

FORGE_ROOT = Path(os.environ.get("EIDOS_FORGE_DIR", str(Path(__file__).resolve().parents[4]))).resolve()
DEFAULT_RUNS_DIR = FORGE_ROOT / "data" / "code_forge" / "ingestion_runs"
ANALYSIS_VERSION = 2
DEFAULT_EXTENSIONS = {".py"}


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
    """Non-destructive ingestion runner (Python-only for v1)."""

    def __init__(
        self,
        db: CodeLibraryDB,
        runs_dir: Path = DEFAULT_RUNS_DIR,
    ) -> None:
        self.db = db
        self.analyzer = CodeAnalyzer()
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

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
        for file_path in root_path.rglob("*"):
            if not file_path.is_file():
                continue
            if any(part in str(file_path) for part in exclude_patterns):
                continue
            if file_path.suffix.lower() in extensions:
                yield file_path

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
                    file_loc = sum(
                        1 for _ in path.open("r", encoding="utf-8", errors="ignore")
                    )
            except OSError:
                continue

            if estimate_loc and seen < sample_size:
                try:
                    lines = sum(
                        1 for _ in path.open("r", encoding="utf-8", errors="ignore")
                    )
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

    def ingest_file(self, file_path: Path, root_path: Path, run_id: str) -> int:
        source_text = file_path.read_text(encoding="utf-8")
        file_hash = self.db.add_text(source_text)
        rel_path = str(file_path.relative_to(root_path))
        abs_path = str(file_path)
        if not self.db.should_process_file(abs_path, file_hash, ANALYSIS_VERSION):
            return 0

        analysis = self.analyzer.analyze_file(file_path)
        if "error" in analysis:
            return 0

        created = 0

        module_meta = analysis.get("module", {})
        module_hash = file_hash

        module_unit = CodeUnit(
            unit_type="module",
            name=file_path.stem,
            qualified_name=file_path.stem,
            file_path=rel_path,
            line_start=module_meta.get("line_start"),
            line_end=module_meta.get("line_end"),
            col_start=module_meta.get("col_start"),
            col_end=module_meta.get("col_end"),
            content_hash=module_hash,
            run_id=run_id,
        )
        module_id = self.db.add_unit(module_unit)
        created += 1

        nodes = list(analysis.get("nodes", []))
        nodes.sort(key=lambda n: (n.get("line_start") or 0, n.get("line_end") or 0))

        id_by_qualified: Dict[str, str] = {file_path.stem: module_id}

        for node in nodes:
            node_source = node.get("source", "")
            node_hash = self.db.add_text(node_source) if node_source else None
            unit_type = node.get("unit_type", "node")
            name = node.get("name", unit_type)
            qualified = node.get("qualified_name") or f"{file_path.stem}.{name}"
            parent_qn = node.get("parent_qualified_name")
            parent_id = id_by_qualified.get(parent_qn, module_id)
            unit = CodeUnit(
                unit_type=unit_type,
                name=name,
                qualified_name=qualified,
                file_path=rel_path,
                line_start=node.get("line_start"),
                line_end=node.get("line_end"),
                col_start=node.get("col_start"),
                col_end=node.get("col_end"),
                content_hash=node_hash,
                parent_id=parent_id,
                run_id=run_id,
                complexity=node.get("complexity"),
            )
            unit_id = self.db.add_unit(unit)
            self.db.add_relationship(parent_id, unit_id, "contains")
            id_by_qualified[qualified] = unit_id
            created += 1

        self.db.update_file_record(abs_path, file_hash, ANALYSIS_VERSION)

        return created

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
        exclude_patterns = exclude_patterns or [".git", "__pycache__", ".venv", "venv", "node_modules"]
        extensions = set(ext.lower() for ext in (extensions or DEFAULT_EXTENSIONS))

        run_id = run_id or uuid.uuid4().hex[:16]
        self.db.create_run(str(root_path), mode, run_id=run_id)
        files: List[str] = []
        files_processed = 0
        units_created = 0
        errors = 0
        files_skipped = 0

        scan_files, total_loc, total_bytes = self._scan_files(
            root_path, extensions, exclude_patterns, max_files=max_files
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
