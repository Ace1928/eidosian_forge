import sys
from .core import mcp
from .routers import system, knowledge, memory

# Import other routers here as they are built...

def main():
    """Entry point for the Eidosian Nexus."""
    print("🔮 Eidosian Nexus Initializing...")
    mcp.run()

if __name__ == "__main__":
    main()