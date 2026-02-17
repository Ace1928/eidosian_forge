# 📊 Viz Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Lens of Eidos.**

> _"To visualize data is to see the patterns of thought."_

## 📊 Overview

`viz_forge` is the visualization engine for Eidosian data. It provides standardized plotting utilities for agent performance, memory density, knowledge graph structures, and system diagnostics. It prioritizes clarity, structural elegance, and "Dark Mode" aesthetics.

## 🏗️ Architecture

- **Plotter Core (`src/viz_forge/`)**: Wraps `matplotlib` and `seaborn` with high-level functions (`line`, `scatter`, `bar`).
- **Eidosian Style**: A curated theme (`eidosian_config.yml`) ensuring consistent colors, fonts, and grid lines across all system artifacts.
- **Auto-Report Generator**: (In Development) Automatically produces PDF or HTML reports from diagnostic metrics.

## 🔗 System Integration

- **Diagnostics Forge**: Feeds time-series data to `viz_forge` for system pulse monitoring.
- **Knowledge Forge**: Uses visualization tools to render the concept graph.

## 🚀 Usage

```python
from viz_forge import Plotter

p = Plotter(theme="elegant")

# Visualize metrics
p.line(
    x=[1, 2, 3, 4], 
    y=[10, 15, 7, 25], 
    title="Inference Latency (ms)",
    label="qwen-1.5b"
)

p.save("reports/viz/latency.png")
```
