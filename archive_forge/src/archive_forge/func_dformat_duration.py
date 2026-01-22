from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Tuple, Dict, Iterator, Any, Type, Set, Iterable, TYPE_CHECKING
from .debug import get_autologger
@classmethod
def dformat_duration(cls, duration: float, pretty: bool=True, short: int=0, include_ms: bool=False, as_int: bool=False) -> Dict[str, Union[float, int]]:
    """
        Formats a duration (secs) into a dict
        """
    if not pretty:
        unit = 'secs' if short else 'seconds'
        value = int(duration) if as_int else duration
        return {unit: value}
    data = {}
    if duration >= 86400:
        unit = 'd' if short > 1 else 'day'
        days = duration // 86400
        if short < 2 and days > 1:
            unit += 's'
        duration -= days * 86400
        data[unit] = int(days) if as_int else days
    if duration >= 3600:
        unit = 'hr' if short else 'hour'
        if short > 1:
            unit = unit[0]
        hours = duration // 3600
        if short < 2 and hours > 1:
            unit += 's'
        duration -= hours * 3600
        data[unit] = int(hours) if as_int else hours
    if duration >= 60:
        unit = 'min' if short else 'minute'
        if short > 1:
            unit = unit[0]
        minutes = duration // 60
        if short < 2 and minutes > 1:
            unit += 's'
        duration -= minutes * 60
        data[unit] = int(minutes) if as_int else minutes
    if duration >= 1:
        unit = 'sec' if short else 'second'
        if short > 1:
            unit = unit[0]
        if short < 2 and duration > 1:
            unit += 's'
        if include_ms:
            seconds = int(duration)
            duration -= seconds
            data[unit] = seconds
        elif short > 1:
            data[unit] = int(duration) if as_int else duration
        else:
            data[unit] = float(f'{duration:.2f}')
    if include_ms and duration > 0:
        unit = 'ms' if short else 'millisecond'
        milliseconds = int(duration * 1000)
        data[unit] = milliseconds
    return data