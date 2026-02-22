# ChatMock Placement Decision (2026-02-22)

## Decision
`ChatMock` remains in `projects/src/chatmock` for now.

## Rationale
- It is currently an incubating proxy/runtime utility rather than a stable Forge-grade subsystem.
- MCP integration only depends on operational behavior, not on `ChatMock` being packaged as a Forge.
- Keeping it in `projects/` avoids premature promotion while workflows, APIs, and auth behavior continue to evolve.

## Promotion Criteria
Promote `ChatMock` into a dedicated Forge only when all criteria are met:
- stable versioned API surface and config contract.
- dedicated CI matrix and security hardening checks pass in Termux + Linux.
- clear ownership boundaries with `llm_forge`/`eidos_mcp` documented.
- migration plan for paths and compatibility shims is approved.

## Current Integration Contract
- MCP remains transport-agnostic and does not require `ChatMock` internals.
- Any usage of `ChatMock` must go through explicit runtime configuration and health checks.
