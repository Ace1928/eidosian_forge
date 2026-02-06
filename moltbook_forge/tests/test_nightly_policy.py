from __future__ import annotations

from moltbook_forge.nightly_policy import NightlyPolicy, NightlyTask, evaluate_task


def test_nightly_policy_blocks_network() -> None:
    policy = NightlyPolicy(allow_network=False)
    task = NightlyTask(name="fetch", description="network fetch", requires_network=True)
    decision = evaluate_task(policy, task)
    assert not decision.allowed
    assert "network_disallowed" in decision.reasons


def test_nightly_policy_requires_receipt() -> None:
    policy = NightlyPolicy()
    task = NightlyTask(name="stage", description="write to staging", requires_write=True)
    decision = evaluate_task(policy, task)
    assert "emit_receipt" in decision.required_actions
    assert "write_to_staging" in decision.required_actions
