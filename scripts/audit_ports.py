from eidosian_core import eidosian
#!/usr/bin/env python3
"""
Port Audit Script for Eidosian Forge.

Checks configuration files and system state to ensure ports adhere to the convention:
Start: 8928
Multiples of 2 (8928, 8930, 8932...)
"""

import os
import sys
import json
import socket
from pathlib import Path

# Convention
START_PORT = 8928
STEP = 2
MAX_PORT = 9000  # Arbitrary upper limit for checking

@eidosian()
def check_port_open(port: int) -> bool:
    """Check if a port is currently in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

@eidosian()
def audit_configs():
    forge_dir = Path("/home/lloyd/eidosian_forge")
    configs = []
    
    # Check known config locations
    locations = [
        forge_dir / "eidos_mcp" / "src" / "eidos_mcp" / "core.py",
        forge_dir / "gis_forge" / "gis_data.json",
        Path("/home/lloyd/.gemini/GEMINI.md"), # Sometimes configs are documented
    ]
    
    print("--- Configuration Audit ---")
    # This is a heuristic text search, not a true parse for Python files
    for loc in locations:
        if loc.exists():
            try:
                content = loc.read_text(encoding='utf-8')
                if "8928" in content:
                    print(f"PASS: Found base port 8928 in {loc.name}")
                else:
                    print(f"INFO: Base port 8928 NOT found in {loc.name} (might be dynamic or implicit)")
            except Exception as e:
                print(f"WARN: Could not read {loc}: {e}")
        else:
            print(f"WARN: Config {loc} not found")

@eidosian()
def scan_ports():
    print("\n--- System Port Scan (Convention: 8928 + 2n) ---")
    found_any = False
    


    for port in range(START_PORT, MAX_PORT, STEP):
        if check_port_open(port):
            print(f"ACTIVE: Port {port} is OPEN.")
            found_any = True
        # also check intermediate ports just in case
        odd_port = port + 1
        if check_port_open(odd_port):
             print(f"ANOMALY: Port {odd_port} (Odd) is OPEN. Violation of 2n step?")
             found_any = True
             
    if not found_any:
        print("INFO: No Eidosian ports (8928-9000) seem to be active on 127.0.0.1.")

if __name__ == "__main__":
    print(f"Auditing Ports. Base: {START_PORT}, Step: {STEP}")
    audit_configs()
    scan_ports()
