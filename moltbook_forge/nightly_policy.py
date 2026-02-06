#!/usr/bin/env python3
"""Nightly build policy enforcement for autonomous tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class NightlyTask:
    name: str
    description: str
    requires_write: bool = False
    requires_network: bool = False
    irreversible: bool = False
    estimated_cost: float = 0.0
    staging_path: Optional[str] = None


@dataclass
class PolicyDecision:
    allowed: bool
    reasons: List[str] = field(default_factory=list)
    required_actions: List[str] = field(default_factory=list)


@dataclass
class NightlyPolicy:
    allow_write: bool = False
    allow_network: bool = False
    allow_irreversible: bool = False
    max_cost: float = 5.0
    staging_only: bool = True
    require_receipt: bool = True


def evaluate_task(policy: NightlyPolicy, task: NightlyTask) -> PolicyDecision:
    reasons: List[str] = []
    required: List[str] = []
    allowed = True

    if task.estimated_cost > policy.max_cost:
        allowed = False
        reasons.append("cost_exceeds_budget")

    if task.requires_network and not policy.allow_network:
        allowed = False
        reasons.append("network_disallowed")

    if task.requires_write and not policy.allow_write:
        allowed = False
        reasons.append("write_disallowed")

    if task.irreversible and not policy.allow_irreversible:
        allowed = False
        reasons.append("irreversible_disallowed")

    if policy.staging_only and task.requires_write:
        required.append("write_to_staging")

    if policy.require_receipt:
        required.append("emit_receipt")

    return PolicyDecision(allowed=allowed, reasons=reasons, required_actions=required)
