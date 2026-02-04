#!/usr/bin/env python3
"""Batch scan and render video files into glyph output."""
from __future__ import annotations

import argparse
from pathlib import Path

from glyph_forge.services.batch import (
    DEFAULT_EXCLUDE_DIRS,
    DEFAULT_EXCLUDE_NAMES,
    BatchState,
    iter_video_files,
    process_videos,
)
from glyph_forge.streaming.naming import VIDEO_EXTS


def _parse_exts(value: str) -> set[str]:
    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    normalized = set()
    for item in items:
        if not item.startswith("."):
            item = "." + item
        normalized.add(item)
    return normalized or {ext.lower() for ext in VIDEO_EXTS}


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch scan and process videos.")
    parser.add_argument("roots", nargs="*", default=["/"], help="Root directories to scan")
    parser.add_argument("--output-dir", help="Directory for output files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--resolution", default="720p", help="Resolution (1080p/720p/480p)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--mode", default="gradient", help="Render mode (gradient/braille)")
    parser.add_argument("--color", default="ansi256", help="Color mode (truecolor/ansi256/none)")
    parser.add_argument("--backend", default="auto", help="Frame backend (auto/cv2/ffmpeg)")
    parser.add_argument("--buffer-seconds", type=float, default=30.0, help="Target buffer duration")
    parser.add_argument("--prebuffer-seconds", type=float, default=5.0, help="Minimum prebuffer duration")
    parser.add_argument("--duration", type=float, default=None, help="Max duration per video (seconds)")
    parser.add_argument("--no-mux-audio", action="store_true", help="Disable audio muxing")
    parser.add_argument(
        "--include-exts",
        default=",".join(sorted({ext.lstrip('.') for ext in VIDEO_EXTS})),
        help="Comma-separated list of video extensions",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=sorted(DEFAULT_EXCLUDE_DIRS),
        help="Directory to exclude (repeatable)",
    )
    parser.add_argument(
        "--exclude-name",
        action="append",
        default=sorted(DEFAULT_EXCLUDE_NAMES),
        help="Directory name to exclude (repeatable)",
    )
    parser.add_argument("--follow-links", action="store_true", help="Follow symlinks while scanning")
    parser.add_argument("--limit", type=int, default=None, help="Max number of videos to process")
    parser.add_argument("--resume", action="store_true", help="Resume from previous state")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume")
    parser.add_argument("--state-dir", default=None, help="State directory for resume logs")
    parser.add_argument("--reset-state", action="store_true", help="Clear previous state before running")
    parser.add_argument("--dry-run", action="store_true", help="Only list matching files, do not process")
    args = parser.parse_args()

    include = _parse_exts(args.include_exts)
    exclude_dirs = {str(Path(p)) for p in args.exclude_dir}
    exclude_names = set(args.exclude_name)
    resume = args.resume and not args.no_resume

    state_path = Path(args.state_dir) if args.state_dir else Path.cwd() / "glyph_forge_output" / "batch_state"
    state = BatchState.load(state_path, reset=args.reset_state)

    def on_error(err: Exception) -> None:
        print(f"Scan warning: {err}")

    def on_progress(path: Path, status: str, error: Exception | None) -> None:
        if status == "dry_run":
            print(f"Found: {path}")
        elif status == "processed":
            print(f"Processed: {path}")
        elif status == "failed":
            print(f"Failed: {path} ({error})")

    videos = iter_video_files(
        args.roots,
        include_exts=include,
        exclude_dirs=exclude_dirs,
        exclude_names=exclude_names,
        follow_links=args.follow_links,
        on_error=on_error,
    )

    processed, failed = process_videos(
        videos,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        resolution=args.resolution,
        fps=args.fps,
        mode=args.mode,
        color=args.color,
        backend=args.backend,
        buffer_seconds=args.buffer_seconds,
        prebuffer_seconds=args.prebuffer_seconds,
        max_duration=args.duration,
        mux_audio=not args.no_mux_audio,
        state=state,
        resume=resume,
        limit=args.limit,
        dry_run=args.dry_run,
        on_progress=on_progress,
    )

    print(f"Batch complete. Processed: {processed} | Failed: {failed} | State: {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
