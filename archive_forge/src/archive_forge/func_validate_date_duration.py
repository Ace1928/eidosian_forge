from __future__ import annotations
from typing import TYPE_CHECKING
from isoduration.formatter.exceptions import DurationFormattingException
def validate_date_duration(date_duration: DateDuration) -> None:
    if date_duration.weeks:
        if date_duration.years or date_duration.months or date_duration.days:
            raise DurationFormattingException('Weeks are incompatible with other date designators')