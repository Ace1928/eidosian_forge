from eidosian_core import eidosian

#!/usr/bin/env python3
"""
Maps .forgengine.json (legacy config) to the central GisCore registry.
Ensures legacy settings are available in the persistent global system.
"""

import json
import logging
import sys
from pathlib import Path

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("map_forgengine")

ROOT_DIR = Path(__file__).resolve().parents[1]
FORGENGINE_JSON = ROOT_DIR / ".forgengine.json"
GIS_DATA_PATH = ROOT_DIR / "eidosian_forge" / "gis_forge" / "gis_data.json"

# Inject sys.path
EIDOSIAN_FORGE_DIR = ROOT_DIR / "eidosian_forge"
if str(EIDOSIAN_FORGE_DIR) not in sys.path:
    sys.path.insert(0, str(EIDOSIAN_FORGE_DIR))

try:
    from gis_forge.gis_core import GisCore
except ImportError as e:
    logger.error(f"Failed to import GisCore: {e}")
    sys.exit(1)


@eidosian()
def main():
    if not FORGENGINE_JSON.exists():
        logger.warning(f"{FORGENGINE_JSON} not found. Skipping.")
        return

    logger.info(f"Reading legacy config from {FORGENGINE_JSON}")
    try:
        legacy_config = json.loads(FORGENGINE_JSON.read_text())
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in .forgengine.json: {e}")
        return

    # Initialize GIS (ensure persistence path is correct)
    # If gis_data.json doesn't exist, GisCore handles it gracefully if path is provided
    # But we want to ensure we are using the 'official' one.
    # Assuming AgentForge/etc use a standard path.
    # Let's check where GisCore is usually instantiated.
    # For now, we use the local one in gis_forge/

    GIS_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    gis = GisCore(persistence_path=GIS_DATA_PATH)

    logger.info(f"Mapping to GisCore at {GIS_DATA_PATH}")

    # Map 'forgengine' key
    gis.set("forgengine", legacy_config)

    # Verify
    mapped = gis.get("forgengine")
    if mapped == legacy_config:
        logger.info("✅ Successfully mapped .forgengine.json to GisCore.")
    else:
        logger.error("❌ Mapping verification failed.")


if __name__ == "__main__":
    main()
