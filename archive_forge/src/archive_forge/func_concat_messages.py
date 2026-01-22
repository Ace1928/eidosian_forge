from __future__ import annotations
import typing as t
from enum import auto
from sqlglot.helper import AutoName
def concat_messages(errors: t.Sequence[t.Any], maximum: int) -> str:
    msg = [str(e) for e in errors[:maximum]]
    remaining = len(errors) - maximum
    if remaining > 0:
        msg.append(f'... and {remaining} more')
    return '\n\n'.join(msg)