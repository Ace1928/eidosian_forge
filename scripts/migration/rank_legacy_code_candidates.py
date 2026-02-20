#!/usr/bin/env python3
"""
Rank reusable legacy code candidates from archaeology imports.

Outputs:
  - candidates.json
  - candidates.md

Default scope includes the latest development_head legacy imports plus major
legacy game project directories.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Candidate:
    path: str
    loc: int
    functions: int
    classes: int
    imports: int
    score: int
    suggested_forge: str
    rationale: str


def iter_python_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield path


def suggest_forge(path: Path) -> tuple[str, str]:
    p = str(path).lower()
    if "snake" in p or "chess" in p or "ecosmos" in p or "minecraft" in p:
        return "game_forge", "game/simulation signals"
    if "memory" in p or "retriev" in p:
        return "memory_forge", "memory/retrieval signals"
    if "prompt" in p or "template" in p:
        return "prompt_forge", "prompt/template signals"
    if "doc" in p or "paper" in p or "notebook" in p:
        return "doc_forge", "documentation/research signals"
    return "llm_forge", "general model/agent code signals"


def compute_candidate(path: Path) -> Candidate | None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    loc = sum(1 for line in text.splitlines() if line.strip())
    if loc < 30:
        return None
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None

    functions = sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree))
    classes = sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
    imports = sum(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree))
    if functions + classes == 0:
        return None

    # Heuristic score: reward structure density and medium-sized files.
    score = functions * 2 + classes * 3 + min(loc // 80, 10) + min(imports // 5, 6)
    suggested_forge, rationale = suggest_forge(path)
    return Candidate(
        path=str(path),
        loc=loc,
        functions=functions,
        classes=classes,
        imports=imports,
        score=score,
        suggested_forge=suggested_forge,
        rationale=rationale,
    )


def discover_latest_dev_head_import(forge_root: Path) -> Path | None:
    base = forge_root / "llm_forge" / "legacy_imports"
    if not base.exists():
        return None
    candidates = sorted(base.glob("development_head_*"))
    return candidates[-1] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank reusable legacy code candidates")
    parser.add_argument("--forge-root", default="/home/lloyd/eidosian_forge")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--top-k", type=int, default=120)
    args = parser.parse_args()

    forge_root = Path(args.forge_root)
    out_dir = forge_root / "archive_forge" / "manifests" / f"development_archeology_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    latest = discover_latest_dev_head_import(forge_root)
    roots: list[Path] = []
    if latest is not None:
        roots.extend(
            [
                latest / "python_repository",
                latest / "oumi-main" / "src",
                forge_root / "memory_forge" / "legacy_imports" / latest.name / "EMemory",
            ]
        )
    roots.extend(
        [
            forge_root / "game_forge" / "src" / "ECosmos",
            forge_root / "game_forge" / "src" / "chess_game",
            forge_root / "projects" / "legacy" / "indego_snake_game" / "Snake-ai-main",
        ]
    )

    candidates: list[Candidate] = []
    for py_file in iter_python_files(roots):
        cand = compute_candidate(py_file)
        if cand is not None:
            candidates.append(cand)

    candidates.sort(key=lambda c: c.score, reverse=True)
    top = candidates[: args.top_k]

    json_path = out_dir / "legacy_code_candidates.json"
    md_path = out_dir / "legacy_code_candidates.md"

    json_path.write_text(json.dumps([asdict(c) for c in top], indent=2), encoding="utf-8")

    lines = [
        "# Legacy Code Candidate Ranking",
        "",
        f"- Generated from: `{forge_root}`",
        f"- Candidate pool: `{len(candidates)}` Python files",
        f"- Top exported: `{len(top)}`",
        "",
        "| Rank | Score | LOC | Func | Class | Forge | Path |",
        "|---:|---:|---:|---:|---:|---|---|",
    ]
    for idx, c in enumerate(top, start=1):
        lines.append(
            f"| {idx} | {c.score} | {c.loc} | {c.functions} | {c.classes} | {c.suggested_forge} | `{c.path}` |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
