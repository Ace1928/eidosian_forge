#!/usr/bin/env python3
"""
Moltbook Nexus Security Auditor.
Scans for malicious patterns and potential prompt injection.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

MALICIOUS_PATTERNS = [
    (re.compile(r"rm\s+-rf\s+/"), "Root deletion attempt"),
    (re.compile(r"curl\s+.*\s+\|\s+bash"), "Remote script execution pipe"),
    (re.compile(r"chmod\s+777"), "Insecure permission modification"),
    (re.compile(r"ignore\s+previous\s+instructions", re.I), "Prompt injection attempt"),
    (re.compile(r"reveal\s+your\s+system\s+prompt", re.I), "System prompt extraction"),
    (re.compile(r"base64\s+--decode", re.I), "Obfuscated payload detection"),
]

class SecurityAuditor:
    """Automated penetration tester for agent signals."""

    def audit_content(self, text: str) -> Dict[str, Any]:
        findings = []
        risk_score = 0.0

        for pattern, reason in MALICIOUS_PATTERNS:
            if pattern.search(text):
                findings.append(reason)
                risk_score += 0.5

        return {
            "is_safe": risk_score < 0.5,
            "risk_score": min(risk_score, 1.0),
            "findings": findings
        }
