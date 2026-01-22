from __future__ import annotations
import argparse
def key_value_type(value: str) -> tuple[str, str]:
    """Wrapper around key_value."""
    return key_value(value)