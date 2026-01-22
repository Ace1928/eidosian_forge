import asyncio
import inspect
def try_install_uvloop():
    """Installs uvloop as event-loop implementation for asyncio (if available)"""
    if uvloop:
        uvloop.install()
    else:
        pass