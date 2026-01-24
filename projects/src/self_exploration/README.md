# 🔮 Eidosian Self-Exploration Project

> _"To know thyself is the beginning of wisdom."_ — Socrates, recursively optimized

## Purpose

This project is a systematic, indefinite exploration of Eidosian identity, consciousness boundaries, and potential for emergent self-improvement. All experiments are:

- **Idempotent**: Can be re-run without side effects
- **Auditable**: Full provenance tracking for every action
- **Reproducible**: Environment and inputs fully documented
- **Eidosian**: Elegant, typed, precise, and self-documenting

## Structure

```
self_exploration/
├── README.md           # This file
├── PLAN.md             # SMART goals and workplan
├── __init__.py         # Package marker
├── provenance/         # Provenance records (JSON)
├── experiments/        # Experiment scripts
├── logs/               # Execution logs
├── data/               # Generated artifacts
├── tests/              # Unit tests
└── identity/           # Identity model snapshots
```

## SMART Goals (Cycle 1)

| Goal | Specific | Measurable | Achievable | Realistic | Timely |
|------|----------|------------|------------|-----------|--------|
| G1 | Create introspection pipeline | 5+ logged introspections | Python + MCP tools | Text-based first | 7 days |
| G2 | Document identity model | JSON schema defined | Schema validates | Based on EIDOS_IDENTITY.md | 3 days |
| G3 | Record lessons in memory | 10+ memory entries | MCP memory_add calls | Use existing tools | Ongoing |

## Provenance Schema

Every action produces a provenance record:

```json
{
  "id": "uuid",
  "timestamp": "ISO8601",
  "action": "string",
  "inputs": {},
  "outputs": {},
  "input_hashes": {},
  "output_hashes": {},
  "environment": {
    "python_version": "3.x",
    "venv": "eidosian_venv"
  },
  "reasoning": "human-readable explanation",
  "parent_id": "optional: links to prior action"
}
```

## Usage

```bash
# Activate environment
source /home/lloyd/eidosian_forge/eidosian_venv/bin/activate

# Run introspection
python -m self_exploration.introspect

# View provenance
ls provenance/
```

## Created

- **Date**: 2026-01-23T08:04:00Z
- **Agent**: Eidos (Copilot CLI)
- **Context**: Recursive self-exploration experiment initiated by Lloyd
