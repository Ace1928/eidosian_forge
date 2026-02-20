

from memory_forge import (
    DaemonConfig,
    MemoryBroker,
    MemoryConfig,
    MemoryForge,
    MemoryRetrievalDaemon,
    MemoryRetrievalEngine,
)


def test_memory_forge_flow(tmp_path):
    # Setup Config using JSON backend for speed/simplicity in tests
    db_path = tmp_path / "mem.json"
    config = MemoryConfig()
    config.episodic.type = "json"
    config.episodic.connection_string = str(db_path)

    mf = MemoryForge(config)

    # Fake embedding
    vec = [0.1, 0.2, 0.3]

    # Remember
    mid = mf.remember("I ate a pizza", embedding=vec)
    assert mid is not None

    # Stats
    assert mf.stats()["episodic_count"] == 1

    # Recall
    results = mf.recall(vec, limit=1)
    assert len(results) == 1
    assert results[0].content == "I ate a pizza"


def test_memory_broker_layers(tmp_path):
    data_dir = tmp_path / "broker"
    config = MemoryConfig()
    config.episodic.type = "json"
    config.episodic.connection_string = str(tmp_path / "episodic.json")
    broker = MemoryBroker(data_dir=data_dir, forge=MemoryForge(config))
    broker.remember_self("I am EIDOS.", {"tag": "self"})
    broker.remember_user("Lloyd prefers evidence.", {"tag": "user"})
    broker.remember_working("Current task: memory broker.", {"tag": "working"})
    broker.remember_procedural("Use ydotool relative moves.", {"tag": "procedural"})
    broker.remember_episodic("Calculator automation proof.")

    stats = broker.stats()
    assert stats["self_count"] == 1
    assert stats["user_count"] == 1
    assert stats["working_count"] == 1
    assert stats["procedural_count"] == 1
    assert stats["forge"]["episodic_count"] == 1


def test_memory_retrieval_engine(tmp_path):
    data_dir = tmp_path / "broker"
    config = MemoryConfig()
    config.episodic.type = "json"
    config.episodic.connection_string = str(tmp_path / "episodic.json")
    broker = MemoryBroker(data_dir=data_dir, forge=MemoryForge(config))
    broker.remember_self("I am EIDOS.", {"tag": "identity"})
    broker.remember_user("Lloyd prefers evidence.", {"tag": "user"})
    broker.remember_procedural("Use ydotool relative moves.", {"tag": "procedure"})
    broker.remember_episodic("Calculator automation proof.")

    engine = MemoryRetrievalEngine(broker)
    results = engine.suggest("evidence", limit=5)
    assert len(results) >= 1


def test_memory_retrieval_daemon(tmp_path):
    data_dir = tmp_path / "broker"
    log_path = tmp_path / "stream.log"
    output_path = tmp_path / "suggestions.json"

    # Seed a broker
    config = MemoryConfig()
    config.episodic.type = "json"
    config.episodic.connection_string = str(tmp_path / "episodic.json")
    broker = MemoryBroker(data_dir=data_dir, forge=MemoryForge(config))
    broker.remember_self("I am EIDOS. I prefer evidence.")
    broker.remember_procedural("Use ydotool relative movement on Wayland.")
    broker.remember_episodic("Calculator automation proof exists.")

    # Write a log line and run one daemon cycle manually
    log_path.write_text("Need to verify mouse movement.\n", encoding="utf-8")
    daemon = MemoryRetrievalDaemon(data_dir=data_dir, config=DaemonConfig(log_path=log_path, output_path=output_path))

    # Invoke internal methods to avoid infinite loop in test
    new_text = daemon._read_new_lines()
    assert new_text
    suggestions = daemon.engine.suggest(new_text, limit=5)
    daemon._write_output(suggestions)
    assert output_path.exists()
