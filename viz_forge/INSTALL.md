# 📊 Viz Forge Installation

Data visualization and charting utilities.

## Quick Install

```bash
pip install -e ./viz_forge

# Verify
python -c "from viz_forge import ChartBuilder; print('✓ Ready')"
```

## CLI Usage

```bash
# Create chart from data
viz-forge chart data.csv --type bar --output chart.png

# Interactive mode
viz-forge interactive

# Help
viz-forge --help
```

## Dependencies

- `matplotlib` - Chart generation
- `eidosian_core` - Universal decorators and logging

