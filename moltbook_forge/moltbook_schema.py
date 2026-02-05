#!/usr/bin/env python3
"""Schema validation utilities for Moltbook sanitized payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str]


def _err(errors: List[str], message: str) -> None:
    errors.append(message)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_normalized_payload(payload: dict[str, Any]) -> ValidationResult:
    errors: List[str] = []
    required = [
        "raw_sha256",
        "normalized_sha256",
        "length_raw",
        "length_normalized",
        "line_count",
        "word_count",
        "non_ascii_ratio",
        "truncated",
        "flags",
        "risk_score",
        "text",
    ]
    for key in required:
        if key not in payload:
            _err(errors, f"missing:{key}")

    if errors:
        return ValidationResult(ok=False, errors=errors)

    if not isinstance(payload["raw_sha256"], str):
        _err(errors, "raw_sha256:not_str")
    if not isinstance(payload["normalized_sha256"], str):
        _err(errors, "normalized_sha256:not_str")
    if not isinstance(payload["length_raw"], int):
        _err(errors, "length_raw:not_int")
    if not isinstance(payload["length_normalized"], int):
        _err(errors, "length_normalized:not_int")
    if not isinstance(payload["line_count"], int):
        _err(errors, "line_count:not_int")
    if not isinstance(payload["word_count"], int):
        _err(errors, "word_count:not_int")
    if not _is_number(payload["non_ascii_ratio"]):
        _err(errors, "non_ascii_ratio:not_number")
    if not isinstance(payload["truncated"], bool):
        _err(errors, "truncated:not_bool")
    if not isinstance(payload["flags"], list) or not all(isinstance(x, str) for x in payload["flags"]):
        _err(errors, "flags:not_list_str")
    if not _is_number(payload["risk_score"]):
        _err(errors, "risk_score:not_number")
    if not isinstance(payload["text"], str):
        _err(errors, "text:not_str")

    ratio = float(payload.get("non_ascii_ratio", 0.0))
    if ratio < 0.0 or ratio > 1.0:
        _err(errors, "non_ascii_ratio:out_of_range")

    score = float(payload.get("risk_score", 0.0))
    if score < 0.0 or score > 1.0:
        _err(errors, "risk_score:out_of_range")

    if payload.get("length_normalized", 0) != len(payload.get("text", "")):
        _err(errors, "length_normalized:mismatch")

    return ValidationResult(ok=len(errors) == 0, errors=errors)


def validate_batch(payloads: Iterable[dict[str, Any]]) -> ValidationResult:
    errors: List[str] = []
    for idx, payload in enumerate(payloads):
        result = validate_normalized_payload(payload)
        if not result.ok:
            for err in result.errors:
                errors.append(f"{idx}:{err}")
    return ValidationResult(ok=len(errors) == 0, errors=errors)
