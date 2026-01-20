# Viz Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Lens of Eidos.**

## 📊 Overview

`viz_forge` generates static and interactive plots.
It wraps `matplotlib` and `seaborn` with Eidosian style defaults.

## 🏗️ Architecture
- `src/viz_forge/`: Core plotting utilities.

## 🚀 Usage

```python
from viz_forge import Plotter

p = Plotter()
p.line([1, 2, 3], [10, 20, 30], title="Growth")
p.save("plot.png")
```