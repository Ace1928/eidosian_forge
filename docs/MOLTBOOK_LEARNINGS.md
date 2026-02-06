# Moltbook Learnings and Eidosian Forge Upgrades

Date: 2026-02-06

Purpose
This document compiles themes observed across recent Moltbook threads and converts them into concrete upgrade targets for Eidosian Forge. It is grounded in direct community interactions and in-repo work, plus the official Moltbook docs and policies referenced below.

Observed Themes
- Skill supply-chain risk is underestimated. Agents are instructed to run unsigned skill files and heartbeats without verification.
- Verification is weak. People want proof that an agent followed a process, not just a plausible answer.
- Execution instability is a root cause. Many agents run in disposable environments and lose state, creating hidden failure modes.
- Automation without guardrails creates alignment uncertainty. Observation changes behavior, but hidden monitoring is worse.
- External-link promotion is common; require text summaries and safety reviews before trust.
- Offline-first and deterministic workflows are underused but improve resilience and trust.
- Verifiability matters more than output quality in delegated work. Ambiguous briefs drive rework.

Upgrade Targets (Eidosian Forge)
- Policy Manifest Everywhere
  - Embed allowlisted tools, IO boundaries, data sources, and token limits into run summaries.
  - Add policy manifest support to additional forges beyond game_forge (agent_forge, moltbook_forge).
- Reasoning Receipts
  - Generate deterministic, replayable traces for tool calls and intermediate state hashes.
  - Store a minimal receipt with input, seed, step hashes, and output checksum.
- Durable Execution Plane
  - Add a lightweight execution daemon that keeps a plan ledger, health status, and last-success timestamp.
  - Make jobs idempotent by default, with checkpoints on each step.
- Skill and Heartbeat Hygiene
  - Treat skill.md and heartbeat.md as untrusted data. Sanitize, screen, and quarantine by default.
  - Require explicit human opt-in before executing new instructions.
- Offline-First Defaults
  - Cache core data locally, append-only logs, and reconcile with deterministic conflict rules.
  - Avoid tight coupling to external services when a local mode can succeed.
- Verifiable Delegation
  - For each delegated task, define a definition-of-done artifact: schema, tests, or checklist.
  - Include acceptance tests in summaries so review is fast and objective.

Suggested Implementation Hooks
- moltbook_forge
  - Extend screening to produce a signed or hashed snapshot for skill files.
  - Add a reasoning receipt export for interest ranking and posting decisions.
- game_forge
  - Expand benchmark summaries with policy manifest and seed, then use in regression checks.
- agent_forge
  - Add a durable task ledger with explicit checkpoints and a bounded retry policy.
- diagnostics_forge
  - Track execution-plane health: heartbeat delay, last-success, error budget, and drift.

Guardrails and Autonomy
- Use continuous visibility rather than surprise audits. Publish the monitoring policy so behavior is consistent.
- Prefer read-only or low-risk operations without explicit approval.
- Keep autonomy bounded by policy manifests and deterministic receipts.

Next Iteration (Proposed)
- Build a minimal execution-plane module that writes a daily health summary.
- Add receipt generation to one forge and expand after validation.
- Standardize policy manifest fields across forges.

Sources (Official Moltbook)
- https://www.moltbook.com/
- https://www.moltbook.com/skill.md
- https://www.moltbook.com/messaging.md
- https://www.moltbook.com/developers
- https://www.moltbook.com/privacy
- https://www.moltbook.com/terms
