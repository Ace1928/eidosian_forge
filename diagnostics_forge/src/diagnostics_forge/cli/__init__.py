from eidosian_core import eidosian
"""
Diagnostics Forge CLI - System diagnostics and health checks.
"""
import argparse
import sys
import os
import platform
import json
from pathlib import Path
from typing import Optional, List

@eidosian()
def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="diagnostics-forge",
        description="🔬 Diagnostics Forge - System diagnostics and health",
    )
    parser.add_argument("--version", action="store_true", help="Show version")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # System info
    subparsers.add_parser("system", help="Show system information")
    
    # Python info
    subparsers.add_parser("python", help="Show Python environment info")
    
    # Memory
    subparsers.add_parser("memory", help="Show memory usage")
    
    # Disk
    disk_parser = subparsers.add_parser("disk", help="Show disk usage")
    disk_parser.add_argument("path", nargs="?", default="/", help="Path to check")
    
    # Health check
    subparsers.add_parser("health", help="Run health checks on forge system")
    
    # Status
    subparsers.add_parser("status", help="Show diagnostics status")
    
    args = parser.parse_args(argv)
    
    if args.version:
        print("Diagnostics Forge v0.1.0")
        return 0
    
    if not args.command:
        parser.print_help()
        return 0
    
    if args.command == "status":
        print("🔬 Diagnostics Forge Status")
        print("  Status: operational")
        return 0
    
    elif args.command == "system":
        info = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "hostname": platform.node(),
        }
        if hasattr(args, 'json') and args.json:
            print(json.dumps(info, indent=2))
        else:
            print("🖥️ System Information")
            for k, v in info.items():
                print(f"  {k}: {v}")
        return 0
    
    elif args.command == "python":
        info = {
            "version": sys.version,
            "executable": sys.executable,
            "prefix": sys.prefix,
            "path_count": len(sys.path),
        }
        if hasattr(args, 'json') and args.json:
            print(json.dumps(info, indent=2))
        else:
            print("🐍 Python Environment")
            for k, v in info.items():
                print(f"  {k}: {v}")
        return 0
    
    elif args.command == "memory":
        try:
            import psutil
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            info = {
                "total_gb": round(mem.total / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "used_percent": mem.percent,
                "swap_total_gb": round(swap.total / (1024**3), 2),
                "swap_used_percent": swap.percent,
            }
            if hasattr(args, 'json') and args.json:
                print(json.dumps(info, indent=2))
            else:
                print("💾 Memory Usage")
                print(f"  RAM: {info['available_gb']}/{info['total_gb']} GB ({info['used_percent']}% used)")
                print(f"  Swap: {info['swap_total_gb']} GB ({info['swap_used_percent']}% used)")
        except ImportError:
            print("Note: Install psutil for detailed memory info")
            print(f"  Python memory: {sys.getsizeof([])} bytes (list object)")
        return 0
    
    elif args.command == "disk":
        try:
            import shutil
            usage = shutil.disk_usage(args.path)
            info = {
                "path": args.path,
                "total_gb": round(usage.total / (1024**3), 2),
                "used_gb": round(usage.used / (1024**3), 2),
                "free_gb": round(usage.free / (1024**3), 2),
                "used_percent": round(100 * usage.used / usage.total, 1),
            }
            if hasattr(args, 'json') and args.json:
                print(json.dumps(info, indent=2))
            else:
                print(f"💿 Disk Usage ({args.path})")
                print(f"  Used: {info['used_gb']}/{info['total_gb']} GB ({info['used_percent']}%)")
                print(f"  Free: {info['free_gb']} GB")
        except Exception as e:
            print(f"Error: {e}")
            return 1
        return 0
    
    elif args.command == "health":
        print("🏥 Health Check")
        checks = []
        
        # Check Python version
        py_ok = sys.version_info >= (3, 10)
        checks.append(("Python >= 3.10", py_ok))
        
        # Check key paths
        forge_root = Path("/home/lloyd/eidosian_forge")
        checks.append(("Forge root exists", forge_root.exists()))
        checks.append(("Venv exists", (forge_root / "eidosian_venv").exists()))
        
        # Print results
        all_ok = True
        for name, ok in checks:
            status = "✓" if ok else "✗"
            print(f"  {status} {name}")
            if not ok:
                all_ok = False
        
        return 0 if all_ok else 1
    
    return 0

app = main

if __name__ == "__main__":
    sys.exit(main())
