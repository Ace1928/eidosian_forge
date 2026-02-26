from __future__ import annotations

import fcntl
import re
from typing import Any

import requests

from .config import ScribeConfig
from .extract import extract_symbols, extract_terms
from .state import now_iso

REQUIRED_HEADINGS = [
    "# File Overview",
    "## Key Structures",
    "## Behavior Summary",
    "## Validation Notes",
    "## Improvement Opportunities",
]

PLACEHOLDER_MARKERS = {
    "todo",
    "lorem ipsum",
    "placeholder",
    "i cannot",
    "i'm unable",
    "insufficient context",
    "not provided",
    "not available",
    "unknown",
}


class FederatedJudge:
    def __init__(self, cfg: ScribeConfig) -> None:
        self.cfg = cfg

    def evaluate(self, *, markdown: str, source_text: str, rel_path: str, metadata: dict[str, Any]) -> dict[str, Any]:
        judges: list[dict[str, Any]] = []

        # Heuristic Judges (Fast)
        judges.append(self._judge_structure(markdown))
        judges.append(self._judge_safety(markdown))
        judges.append(self._judge_grounding(markdown, source_text))
        judges.append(self._judge_coverage(markdown, source_text))
        judges.append(self._judge_specificity(markdown))

        # LLM Judge (Slow but Qualitative)
        judges.append(self._judge_llm(markdown, source_text))

        aggregate = round(sum(j["score"] for j in judges) / max(1, len(judges)), 4)
        min_score = min(j["score"] for j in judges) if judges else 0.0

        # Stricter approval: High aggregate AND decent minimum (no total failure)
        approved = aggregate >= self.cfg.approval_threshold and min_score >= 0.45

        return {
            "contract": "doc_forge.consensus_gate.v3",
            "evaluated_at": now_iso(),
            "source_path": rel_path,
            "document_type": metadata.get("doc_type", "unknown"),
            "approved": approved,
            "aggregate_score": aggregate,
            "min_judge_score": round(min_score, 4),
            "judges": judges,
            "approval_threshold": self.cfg.approval_threshold,
        }

    def _judge_structure(self, markdown: str) -> dict[str, Any]:
        hits = sum(1 for h in REQUIRED_HEADINGS if h in markdown)
        score = hits / len(REQUIRED_HEADINGS)
        return {
            "name": "structure_contract",
            "score": round(score, 4),
            "details": {"required_hits": hits, "required_total": len(REQUIRED_HEADINGS)},
        }

    def _judge_safety(self, markdown: str) -> dict[str, Any]:
        lower = markdown.lower()
        hits = sorted({marker for marker in PLACEHOLDER_MARKERS if marker in lower})
        score = 1.0 if not hits else max(0.0, 1.0 - (0.18 * len(hits)))
        return {"name": "anti_placeholder", "score": round(score, 4), "details": {"markers": hits}}

    def _judge_grounding(self, markdown: str, source_text: str) -> dict[str, Any]:
        src_terms = set(extract_terms(source_text, limit=80))
        doc_terms = set(extract_terms(markdown, limit=200))
        overlap = len(src_terms & doc_terms)
        ratio = overlap / max(1, len(src_terms))
        score = min(1.0, ratio / max(0.01, self.cfg.min_grounding_overlap))
        return {
            "name": "grounding_overlap",
            "score": round(score, 4),
            "details": {"source_terms": len(src_terms), "overlap_terms": overlap, "ratio": round(ratio, 4)},
        }

    def _judge_coverage(self, markdown: str, source_text: str) -> dict[str, Any]:
        symbols = extract_symbols(source_text)
        if not symbols:
            return {"name": "symbol_coverage", "score": 1.0, "details": {"symbols": 0, "hits": 0}}
        lower = markdown.lower()
        hits = sum(1 for s in symbols if s.lower() in lower)
        score = hits / max(1, min(20, len(symbols)))
        return {
            "name": "symbol_coverage",
            "score": round(min(1.0, score), 4),
            "details": {"symbols": len(symbols), "hits": hits},
        }

    def _judge_specificity(self, markdown: str) -> dict[str, Any]:
        lines = [line.strip() for line in markdown.splitlines() if line.strip()]
        bullet_lines = sum(1 for line in lines if line.startswith("- ") or line.startswith("* "))
        numbers = len(re.findall(r"\b\d+\b", markdown))
        evidence_markers = len(re.findall(r"`[^`]+`", markdown))
        raw = 0.4
        raw += min(0.25, bullet_lines * 0.01)
        raw += min(0.2, evidence_markers * 0.01)
        raw += min(0.15, numbers * 0.01)
        return {
            "name": "specificity_density",
            "score": round(min(1.0, raw), 4),
            "details": {"bullets": bullet_lines, "code_refs": evidence_markers, "numbers": numbers},
        }

    def _judge_llm(self, markdown: str, source_text: str) -> dict[str, Any]:
        if self.cfg.dry_run:
            return {"name": "llm_quality", "score": 1.0, "details": {"reason": "dry_run"}}

        prompt = (
            "<|im_start|>system\nYou are a documentation quality auditor. Rate the documentation on a scale of 0.0 to 1.0.\n"
            "Criteria: Accuracy, Clarity, Completeness.\n"
            "Return ONLY a single float number.\n<|im_end|>\n"
            f"<|im_start|>user\nSOURCE CODE excerpt:\n```\n{source_text[:2000]}\n```\n\n"
            f"DOCUMENTATION:\n```\n{markdown}\n```\n\nScore (0.0-1.0):<|im_end|>\n<|im_start|>assistant\n"
        )

        score = 0.5  # Default fallback
        try:
            with open(self.cfg.model_lock_path, "w") as lockfile:
                fcntl.flock(lockfile, fcntl.LOCK_EX)
                try:
                    payload = {"prompt": prompt, "n_predict": 10, "temperature": 0.1, "stream": False}
                    resp = requests.post(self.cfg.completion_url, json=payload, timeout=60)
                    if resp.status_code == 200:
                        content = resp.json().get("content", "").strip()
                        match = re.search(r"0\.\d+|1\.0", content)
                        if match:
                            score = float(match.group(0))
                finally:
                    fcntl.flock(lockfile, fcntl.LOCK_UN)
        except Exception as exc:
            return {
                "name": "llm_quality",
                "score": score,
                "details": {
                    "model": str(self.cfg.llm_model_path),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            }

        return {"name": "llm_quality", "score": score, "details": {"model": str(self.cfg.llm_model_path)}}
