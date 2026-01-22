from typing import Callable, Optional
def set_on_start(self, on_start: Optional[Callable[['TrackedActor'], None]]):
    self._on_start = on_start