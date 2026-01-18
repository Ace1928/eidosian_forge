import asyncio

from core.event_bus import EventBus


def test_publish_and_receive() -> None:
    bus = EventBus()
    received: list[int] = []

    async def handler(data: int) -> None:
        received.append(data)

    async def run() -> None:
        await bus.subscribe("num", handler)
        await bus.publish("num", 5)
        await asyncio.sleep(0)  # allow tasks to complete

    asyncio.run(run())
    assert received == [5]


def test_unsubscribe() -> None:
    bus = EventBus()
    received: list[str] = []

    async def handler(data: str) -> None:
        received.append(data)

    async def run() -> None:
        await bus.subscribe("msg", handler)
        await bus.unsubscribe("msg", handler)
        await bus.publish("msg", "hi")
        await asyncio.sleep(0)

    asyncio.run(run())
    assert received == []


def test_multiple_subscribers() -> None:
    bus = EventBus()
    order: list[str] = []

    async def a(data: str) -> None:
        order.append("a")

    async def b(data: str) -> None:
        order.append("b")

    async def run() -> None:
        await bus.subscribe("note", a)
        await bus.subscribe("note", b)
        await bus.publish("note", "x")
        await asyncio.sleep(0)

    asyncio.run(run())
    assert sorted(order) == ["a", "b"]
