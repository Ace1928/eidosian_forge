import concurrent.futures
import fnmatch
import grp
import hashlib
import json
import logging
import pwd
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from eidosian_core import eidosian
from tqdm import tqdm

from .context_summarizer import SummarizationError, Summarizer
from .context_utils import CONTEXT_DIR

HOME = Path.home().resolve()


class CatalogBuilder:
    def __init__(self, config: Dict, logger: logging.Logger, use_codex: bool = False):
        self.config = config
        self.logger = logger
        self.catalog_cfg = config.get("catalog", {})
        summary_cfg = self.catalog_cfg.get("summary", {})
        cache_path = summary_cfg.get("cache_path")
        self.summarizer = Summarizer(
            model=summary_cfg.get("model", "qwen2.5:1.5b-Instruct"),  # Updated default model
            max_chars=summary_cfg.get("max_chars", 3200),
            min_chars=summary_cfg.get("min_chars", 512),
            timeout_seconds=summary_cfg.get("timeout_seconds", 60),
            cache_path=(Path(cache_path) if cache_path else CONTEXT_DIR / "summary_cache.json"),
            logger=logger,
            use_codex=use_codex,
        )
        self.ignore_patterns = summary_cfg.get("ignore_patterns") or []
        self.ignore_prefixes = summary_cfg.get("ignore_prefixes") or []
        self.roots = self.catalog_cfg.get("roots") or []
        self.include_extensions = tuple(summary_cfg.get("include_extensions") or [])
        self.profile = {
            "directories": 0,
            "files": 0,
            "summary_tasks": 0,
            "summary_duration": 0.0,
            "start": time.perf_counter(),
        }
        self.tasks: List[tuple] = []

    def _hash(self, path: Path) -> str | None:
        algorithm = self.catalog_cfg.get("hash_algorithm", "md5")
        try:
            digest = hashlib.new(algorithm)
        except ValueError:
            digest = hashlib.md5()
        try:
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(8192), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except OSError as exc:
            self.logger.debug("Unable to hash %s: %s", path, exc)
            return None

    def _read_snippet(self, path: Path) -> str:
        max_chars = self.catalog_cfg.get("summary", {}).get("max_chars", 3200)
        try:
            with path.open("rb") as fh:
                raw = fh.read(max_chars)
        except OSError as exc:
            self.logger.debug("Skipping snippet for %s: %s", path, exc)
            return ""
        if b"\x00" in raw:
            return ""
        return raw.decode("utf-8", errors="replace").strip()

    def _should_summarize(self, path: Path, rel_path: str) -> bool:
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(path.name, pattern):
                return False
            if fnmatch.fnmatch(rel_path, pattern):
                return False
        for prefix in self.ignore_prefixes:
            if rel_path.startswith(prefix):
                return False
        if "/.git/" in rel_path or rel_path.startswith(".git/"):
            return False
        if self.include_extensions:
            name = path.name.lower()
            if not any(name.endswith(ext.lower()) for ext in self.include_extensions):
                return False
        return True

    def _manual_note_for(self, rel_path: str, name: str) -> Dict:
        manual = self.config.get("manual_notes", {})
        return manual.get(rel_path) or manual.get(name) or {}

    def _describe(self, path: Path, depth: int, max_files: int) -> Dict:
        rel_path = "." if path == HOME else str(path.relative_to(HOME))
        entry = {
            "name": path.name or rel_path,
            "relative_path": rel_path,
            "absolute_path": str(path.resolve()),
            "depth": depth,
            "manual": self._manual_note_for(rel_path, path.name),
            "statistics": {},
        }
        try:
            stats = path.lstat()
        except OSError as exc:
            entry["errors"] = [str(exc)]
            self.logger.debug("Failed to stat %s: %s", path, exc)
            return entry

        entry["statistics"].update(
            {
                "size_bytes": stats.st_size,
                "last_modified": datetime.fromtimestamp(stats.st_mtime, tz=timezone.utc).isoformat(),
                "mode": oct(stats.st_mode & 0o777),
                "is_symlink": stat.S_ISLNK(stats.st_mode),
                "owner": None,
                "group": None,
            }
        )
        try:
            entry["statistics"]["owner"] = pwd.getpwuid(stats.st_uid).pw_name
        except KeyError:
            entry["statistics"]["owner"] = stats.st_uid
        try:
            entry["statistics"]["group"] = grp.getgrgid(stats.st_gid).gr_name
        except KeyError:
            entry["statistics"]["group"] = stats.st_gid

        is_dir = stat.S_ISDIR(stats.st_mode)
        is_file = stat.S_ISREG(stats.st_mode)
        entry["type"] = "directory" if is_dir else "file" if is_file else "other"
        if is_dir:
            self.profile["directories"] += 1
            entry["children"] = []
            if depth >= self.catalog_cfg.get("scan_depth", 3):
                entry["note"] = "Maximum depth reached; children not expanded."
                return entry
            try:
                children = sorted(path.iterdir(), key=lambda x: x.name.lower())
            except OSError as exc:
                entry["errors"] = entry.get("errors", []) + [str(exc)]
                return entry
            trimmed = 0
            for child in children:
                if child.name.startswith("."):
                    continue
                if trimmed >= max_files:
                    entry["note"] = f"Directory limit ({max_files}) reached."
                    break
                entry["children"].append(self._describe(child, depth + 1, max_files))
                trimmed += 1
            entry["child_count"] = len(entry["children"])
        elif is_file:
            self.profile["files"] += 1
            entry["hash"] = self._hash(path)
            snippet = self._read_snippet(path)
            if snippet and self._should_summarize(path, rel_path):
                self.tasks.append((entry, snippet, path))
        return entry

    def _summarize_tasks(self):
        concurrency = self.catalog_cfg.get("summary", {}).get("concurrency", 3)
        if not self.tasks:
            return
        self.profile["summary_tasks"] = len(self.tasks)

        # tqdm progress bar
        pbar = tqdm(total=len(self.tasks), desc="Summarizing", unit="file")

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_to_entry = {
                    executor.submit(self.summarizer.summarize, snippet, path): (entry, path)
                    for entry, snippet, path in self.tasks
                }
                for future in concurrent.futures.as_completed(future_to_entry):
                    entry, path = future_to_entry[future]
                    try:
                        result = future.result()
                    except SummarizationError as exc:
                        self.logger.warning("Summary failed for %s: %s", path, exc)
                        entry["summary_error"] = str(exc)
                        pbar.update(1)
                        continue
                    except Exception as exc:
                        self.logger.error("Unexpected error for %s: %s", path, exc)
                        pbar.update(1)
                        continue

                    entry["summary"] = result["summary"]
                    entry["summary_meta"] = result["meta"]
                    self.profile["summary_duration"] += result["meta"]["duration_sec"]
                    pbar.update(1)

                    # Incremental save (optional, but good for safety)
                    if pbar.n % 10 == 0:
                        self.summarizer.save_cache()

        except KeyboardInterrupt:
            self.logger.warning("\nCatalog generation interrupted by user.")
            pbar.close()
            print("\nStopping... Saving progress...")
            self.summarizer.save_cache()
            print("Summary cache saved. You can resume later.")
            sys.exit(130)
        finally:
            pbar.close()
            # Final save
            self.summarizer.save_cache()

    @eidosian()
    def build(self) -> Dict:
        max_files = self.catalog_cfg.get("max_files_per_dir", 80)
        entries = []
        for rel in self.roots:
            candidate = Path(rel)
            if candidate.is_absolute():
                path = candidate
            else:
                path = HOME / rel
            if not path.exists():
                self.logger.debug("Skipping missing catalog root %s", path)
                continue
            entries.append(self._describe(path, 0, max_files))
        self._summarize_tasks()

        self.profile["total_duration_sec"] = time.perf_counter() - self.profile["start"]
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "root": str(HOME),
            "catalog_config": self.catalog_cfg,
            "entries": entries,
            "summary_records": self.summarizer.records,
            "profiling": {
                "directories": self.profile["directories"],
                "files": self.profile["files"],
                "summary_tasks": self.profile["summary_tasks"],
                "summary_duration": self.profile["summary_duration"],
                "total_duration_sec": self.profile["total_duration_sec"],
            },
        }
        return payload


@eidosian()
def generate_catalog(config: Dict, logger: logging.Logger, output_path: Path, use_codex: bool = False):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    builder = CatalogBuilder(config, logger, use_codex=use_codex)
    try:
        payload = builder.build()
        with output_path.open("w") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        return payload
    except KeyboardInterrupt:
        # Catch interrupt during build (e.g. dir scanning)
        print("\nCatalog generation interrupted. Saving what we have...")
        # We can't easily save a partial catalog structure here without more refactoring,
        # but the summary cache is saved by builder.summarizer
        sys.exit(130)
