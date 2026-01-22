from time import time  # noqa: F401
from typing import TYPE_CHECKING, Any, List, Tuple
def get_engine_status(engine: 'ExecutionEngine') -> List[Tuple[str, Any]]:
    """Return a report of the current engine status"""
    tests = ['time()-engine.start_time', 'len(engine.downloader.active)', 'engine.scraper.is_idle()', 'engine.spider.name', 'engine.spider_is_idle()', 'engine.slot.closing', 'len(engine.slot.inprogress)', 'len(engine.slot.scheduler.dqs or [])', 'len(engine.slot.scheduler.mqs)', 'len(engine.scraper.slot.queue)', 'len(engine.scraper.slot.active)', 'engine.scraper.slot.active_size', 'engine.scraper.slot.itemproc_size', 'engine.scraper.slot.needs_backout()']
    checks: List[Tuple[str, Any]] = []
    for test in tests:
        try:
            checks += [(test, eval(test))]
        except Exception as e:
            checks += [(test, f'{type(e).__name__} (exception)')]
    return checks