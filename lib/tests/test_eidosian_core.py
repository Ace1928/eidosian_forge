from __future__ import annotations

import asyncio

from eidosian_core import (
    Benchmark,
    EidosianConfig,
    Profiler,
    Tracer,
    benchmark_context,
    eidosian,
    get_config,
    profile_context,
    set_config,
    trace_context,
)


def test_eidosian_sync_decorator_roundtrip() -> None:
    @eidosian(log=False, profile=False, benchmark=False, trace=False)
    def add(x: int, y: int) -> int:
        return x + y

    assert add(2, 3) == 5
    assert getattr(add, "__eidosian_decorated__", False) is True


def test_eidosian_async_decorator_roundtrip() -> None:
    @eidosian(log=False, profile=False, benchmark=False, trace=False)
    async def add(x: int, y: int) -> int:
        return x + y

    assert asyncio.run(add(4, 5)) == 9
    assert getattr(add, "__eidosian_decorated__", False) is True


def test_config_roundtrip() -> None:
    cfg = EidosianConfig()
    cfg.logging.enabled = False
    cfg.benchmarking.enabled = True
    cfg.benchmarking.iterations = 3

    raw = cfg.to_json()
    loaded = EidosianConfig.from_json(raw)

    assert loaded.logging.enabled is False
    assert loaded.benchmarking.enabled is True
    assert loaded.benchmarking.iterations == 3


def test_set_and_get_global_config() -> None:
    cfg = EidosianConfig()
    cfg.logging.level = "DEBUG"
    set_config(cfg)
    current = get_config()
    assert current.logging.level == "DEBUG"


def test_profiler_generates_report() -> None:
    profiler = Profiler(top_n=5)
    profiler.start()
    total = sum(range(10000))
    report = profiler.stop("sum_range")

    assert total > 0
    assert report.function_name == "sum_range"
    assert report.total_time >= 0.0


def test_benchmark_generates_statistics() -> None:
    bench = Benchmark(iterations=3, warmup=1)

    def work() -> int:
        return sum(range(5000))

    result = bench.run(work, name="work")
    assert result.name == "work"
    assert result.error is None
    assert result.iterations == 3
    assert len(result.times) == 3
    assert result.mean_time >= 0.0


def test_tracer_hierarchy() -> None:
    tracer = Tracer(capture_args=True, capture_result=True)
    tracer.start_span("root", args=(1,), kwargs={"x": 2})
    tracer.start_span("child")
    tracer.end_span(result="child-ok")
    tracer.end_span(result="root-ok")

    assert len(tracer.root_spans) == 1
    root = tracer.root_spans[0]
    assert root.name == "root"
    assert len(root.children) == 1
    assert root.children[0].name == "child"


def test_context_managers_smoke() -> None:
    with profile_context("profile-smoke", print_report=False):
        sum(range(1000))

    with benchmark_context("benchmark-smoke", iterations=1, print_result=False) as result:
        sum(range(1000))

    assert result.name == "benchmark-smoke"

    with trace_context("trace-smoke", print_result=False) as tracer:
        sum(range(100))

    assert tracer.root_spans
