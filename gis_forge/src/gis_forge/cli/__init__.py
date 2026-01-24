"""
GIS Forge CLI - Geographic information system utilities.
"""
import argparse
import sys
import json
from typing import Optional, List
from eidosian_core import eidosian

@eidosian()
def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="gis-forge",
        description="🌍 GIS Forge - Geographic utilities",
    )
    parser.add_argument("--version", action="store_true", help="Show version")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Distance
    dist_parser = subparsers.add_parser("distance", help="Calculate distance between points")
    dist_parser.add_argument("lat1", type=float, help="Latitude 1")
    dist_parser.add_argument("lon1", type=float, help="Longitude 1")
    dist_parser.add_argument("lat2", type=float, help="Latitude 2")
    dist_parser.add_argument("lon2", type=float, help="Longitude 2")
    
    # Geocode
    geo_parser = subparsers.add_parser("geocode", help="Geocode an address")
    geo_parser.add_argument("address", help="Address to geocode")
    
    # Status
    subparsers.add_parser("status", help="Show GIS Forge status")
    
    args = parser.parse_args(argv)
    
    if args.version:
        print("GIS Forge v0.1.0")
        return 0
    
    if not args.command:
        parser.print_help()
        return 0
    
    if args.command == "status":
        print("🌍 GIS Forge Status")
        print("  Status: operational")
        return 0
    
    elif args.command == "distance":
        from math import radians, sin, cos, sqrt, atan2
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(radians, [args.lat1, args.lon1, args.lat2, args.lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = R * c
        
        print(f"Distance: {distance:.2f} km")
        return 0
    
    return 0

app = main

if __name__ == "__main__":
    sys.exit(main())
