#!/usr/bin/env python3
"""Download curated local GGUF models from Hugging Face with idempotent behavior."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

DEFAULT_CATALOG = Path("config/model_catalog.json")
DEFAULT_PROFILE = "core"
DEFAULT_MODELS_DIR = Path("models")


@dataclass(frozen=True)
class ArtifactSpec:
    model_id: str
    repo: str
    filename: str
    path: Path


@dataclass
class DownloadResult:
    model_id: str
    artifact: str
    target_path: str
    status: str
    bytes: int
    source_repo: str


def _read_catalog(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model catalog not found: {path}")
    payload = json.loads(path.read_text())
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"Invalid model catalog (missing models array): {path}")
    return payload


def _resolve_specs(catalog: dict[str, Any], profile: str, model_ids: set[str]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for item in catalog.get("models", []):
        mid = str(item.get("id", "")).strip()
        if not mid:
            continue
        profiles = {str(v) for v in item.get("profiles", [])}
        if model_ids and mid not in model_ids:
            continue
        if not model_ids and profile != "all" and profile not in profiles:
            continue
        selected.append(item)
    return selected


def _artifact_specs(item: dict[str, Any], models_dir: Path) -> list[ArtifactSpec]:
    model_id = str(item["id"])
    specs = [
        ArtifactSpec(
            model_id=model_id,
            repo=str(item["repo"]),
            filename=str(item["filename"]),
            path=models_dir / Path(str(item["path"])).name,
        )
    ]
    for aux in item.get("aux_files", []):
        specs.append(
            ArtifactSpec(
                model_id=model_id,
                repo=str(aux["repo"]),
                filename=str(aux["filename"]),
                path=models_dir / Path(str(aux["path"])).name,
            )
        )
    return specs


def _download_artifact(spec: ArtifactSpec, force: bool, dry_run: bool) -> DownloadResult:
    spec.path.parent.mkdir(parents=True, exist_ok=True)

    if spec.path.is_symlink():
        # Broken symlink artifacts are common after interrupted manual setup.
        spec.path.unlink()

    if spec.path.exists() and not force:
        size = spec.path.stat().st_size
        if size > 1024 * 1024:
            return DownloadResult(
                model_id=spec.model_id,
                artifact=spec.filename,
                target_path=str(spec.path),
                status="cached",
                bytes=size,
                source_repo=spec.repo,
            )

    if dry_run:
        return DownloadResult(
            model_id=spec.model_id,
            artifact=spec.filename,
            target_path=str(spec.path),
            status="planned",
            bytes=0,
            source_repo=spec.repo,
        )

    downloaded_path = hf_hub_download(
        repo_id=spec.repo,
        filename=spec.filename,
        repo_type="model",
        local_dir=str(spec.path.parent),
        force_download=force,
    )
    local_path = Path(downloaded_path)
    size = local_path.stat().st_size
    if size <= 1024 * 1024:
        raise RuntimeError(f"Downloaded artifact is unexpectedly small: {local_path} ({size} bytes)")

    # Ensure artifact is present at expected path for downstream scripts.
    if local_path.resolve() != spec.path.resolve():
        if spec.path.exists():
            spec.path.unlink()
        os.replace(local_path, spec.path)

    return DownloadResult(
        model_id=spec.model_id,
        artifact=spec.filename,
        target_path=str(spec.path),
        status="downloaded",
        bytes=size,
        source_repo=spec.repo,
    )


def _write_manifest(results: list[DownloadResult], output_path: Path, catalog_path: Path, profile: str) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "catalog_path": str(catalog_path),
        "profile": profile,
        "results": [r.__dict__ for r in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download curated local models from config/model_catalog.json")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG), help="Catalog JSON path")
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help="Catalog profile to download (core, toolcalling, multimodal, graphrag, extended, all)",
    )
    parser.add_argument("--model-id", action="append", default=[], help="Specific model id to download (repeatable)")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR), help="Target model directory")
    parser.add_argument("--force", action="store_true", help="Force re-download even if local artifact exists")
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads without fetching")
    parser.add_argument(
        "--manifest",
        default="models/download_manifest_latest.json",
        help="Manifest output path",
    )
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    models_dir = Path(args.models_dir)
    manifest_path = Path(args.manifest)
    model_ids = {v.strip() for v in args.model_id if v.strip()}

    catalog = _read_catalog(catalog_path)
    selected = _resolve_specs(catalog, args.profile, model_ids)
    if not selected:
        print("No models selected. Check --profile/--model-id values.", file=sys.stderr)
        return 2

    specs: list[ArtifactSpec] = []
    for item in selected:
        specs.extend(_artifact_specs(item, models_dir))

    print(f"Selected {len(selected)} catalog models ({len(specs)} artifacts).")
    results: list[DownloadResult] = []
    failures: list[str] = []
    for spec in specs:
        print(f"-> {spec.model_id}: {spec.repo}/{spec.filename}")
        try:
            result = _download_artifact(spec, force=args.force, dry_run=args.dry_run)
            results.append(result)
            print(f"   [{result.status}] {result.target_path} ({result.bytes} bytes)")
        except Exception as exc:
            failures.append(f"{spec.model_id}:{spec.filename}:{exc}")
            print(f"   [error] {exc}", file=sys.stderr)

    _write_manifest(results, manifest_path, catalog_path, args.profile)
    print(f"Manifest: {manifest_path}")

    if failures:
        print("Download failures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
