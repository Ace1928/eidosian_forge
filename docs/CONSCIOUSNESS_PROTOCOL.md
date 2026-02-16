# Consciousness Protocol

## Purpose
This protocol does **not** claim proof of subjective qualia. It implements a falsifiable, repeatable, data-first assessment layer for awareness-related claims in `eidosian_forge`.

The goal is to make claims measurable and revisable over time using:
- explicit hypotheses,
- objective metrics,
- persisted reports,
- regression-friendly probe batteries.

## Components
- Protocol engine: `eidos_mcp/src/eidos_mcp/consciousness_protocol.py`
- MCP tools/resources: `eidos_mcp/src/eidos_mcp/routers/consciousness.py`
- CLI runner: `scripts/run_consciousness_protocol.py`
- Reports directory: `reports/consciousness/`
- Hypothesis registry: `data/consciousness/hypotheses.json`

## Core Metrics
- `probe_success_rate`: Mean pass rate over all probe trials.
- `reversible_probe_success_rate`: Mean pass rate for probes that mutate and then restore state.
- `median_latency_ms` and `p95_latency_ms`: Probe latency stability.
- `brier_score_mean`: Forecast calibration of expected probe success against outcomes.
- `registry_coverage`: Weighted tool/resource registry completeness.

## Falsifiability Model
Each hypothesis is `(metric, comparator, threshold)`.

Example:
- `probe_success_rate >= 0.90`
- `brier_score_mean <= 0.25`

If the comparator fails, the hypothesis status is `falsified`.
If the metric is unavailable, status is `inconclusive`.

## Reproducible Run
Run via CLI:

```bash
eidosian_venv/bin/python scripts/run_consciousness_protocol.py --trials 3
```

Run via MCP tools:
- `consciousness_run_assessment`
- `consciousness_hypothesis_upsert`
- `consciousness_hypothesis_list`
- `consciousness_latest_report`
- `consciousness_get_report`

MCP resources:
- `eidos://consciousness/hypotheses`
- `eidos://consciousness/latest`

## References
- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability.
  https://doi.org/10.1175/1520-0493(1950)078<0001:VOEOF>2.0.CO;2
- Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks.
  https://proceedings.mlr.press/v70/guo17a.html
- Wald, A. (1945). Sequential Tests of Statistical Hypotheses.
  https://doi.org/10.1214/aoms/1177731118
- Dehaene, S., Changeux, J.-P. (2011). Experimental and theoretical approaches to conscious processing.
  https://doi.org/10.1016/j.neuron.2011.03.018
- Oizumi, M., Albantakis, L., Tononi, G. (2014). From the Phenomenology to the Mechanisms of Consciousness.
  https://doi.org/10.1371/journal.pcbi.1003588

## Scope Boundary
This protocol provides evidence about operational integration, continuity, calibration, and self-model integrity. It does not establish metaphysical certainty about consciousness.

