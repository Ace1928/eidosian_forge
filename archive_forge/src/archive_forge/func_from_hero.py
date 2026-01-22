import typing
from typing import Any
from minerl.herobraine.hero import spaces
from minerl.herobraine.hero.handlers.translation import TranslationHandler
def from_hero(self, x: typing.Dict[str, Any]):
    return x['isGuiOpen']