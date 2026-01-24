# 🗺️ GIS Forge Installation

Geographic Information System utilities and mapping.

## Quick Install

```bash
pip install -e ./gis_forge

# Verify
python -c "from gis_forge import GeoProcessor; print('✓ Ready')"
```

## CLI Usage

```bash
# Process shapefile
gis-forge process input.shp --output output.geojson

# Geocode address
gis-forge geocode "123 Main St, City"

# Help
gis-forge --help
```

## Dependencies

- `geopandas` - Geographic data
- `shapely` - Geometric operations
- `eidosian_core` - Universal decorators and logging

