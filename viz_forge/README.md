# 📊 Viz Forge ⚡

> _"The Lens of Eidos. To visualize data is to see the patterns of thought."_

## 🧠 Overview

`viz_forge` is the central visualization and rendering engine for Eidosian data. It provides standardized plotting utilities for agent performance, memory density, knowledge graph structures, and system diagnostics. It prioritizes clarity, structural elegance, and strict adherence to the "Eidosian Aesthetic" (Dark Mode, high contrast, minimal noise).

```ascii
      ╭───────────────────────────────────────────╮
      │                VIZ FORGE                  │
      │    < Metrics | Graphs | Heatmaps >        │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │   PLOTTER ENGINE    │   │  STYLE CONFIG   │
      │ (Matplotlib Wrap)   │   │ (Eidosian Core) │
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: Data Visualization & Reporting
- **Test Coverage**: Core plotting logic verified.
- **Architecture**:
  - `src/viz_forge/cli/`: Minimal entry points.
  - `eidosian_config.yml`: Unified configuration dictating exact color hex codes and grid alphas.

## 🚀 Usage & Workflows

### Python API

```python
from viz_forge import Plotter

# Initialize plotter with the standard Eidosian style
p = Plotter(theme="elegant")

# Generate standard telemetry visual
p.line(
    x=[1, 2, 3, 4], 
    y=[10, 15, 7, 25], 
    title="Inference Latency (ms)",
    label="qwen-1.5b"
)

# Render to standard reports directory
p.save("reports/viz/latency.png")
```

## 🔗 System Integration

- **Diagnostics Forge**: Feeds time-series operational pulse data to `viz_forge` for rendering system health dashboards.
- **Knowledge Forge**: Uses advanced layout algorithms to render 2D projections of the semantic concept graph.

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy docs into unified Eidosian standard.
- [ ] Add explicit support for `seaborn` statistical distribution wrappers.

### Future Vector (Phase 3+)
- Build an Auto-Report Generator that automatically pulls latest JSON benchmarks from `code_forge` and `llm_forge`, and renders a composite PDF or interactive HTML report via `article_forge`.

---
*Generated and maintained by Eidos.*
