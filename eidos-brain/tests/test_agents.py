import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import Agent, UtilityAgent, ExperimentAgent


def test_utility_agent_perform_task():
    agent = UtilityAgent()
    result = agent.perform_task("clean")
    assert result == "Performed clean"


def test_utility_agent_batch_perform():
    agent = UtilityAgent()
    results = agent.batch_perform(["clean", "build"])
    assert results == ["Performed clean", "Performed build"]


def test_agents_share_base_class():
    """Verify that agents derive from :class:`Agent`."""
    assert isinstance(UtilityAgent(), Agent)
    assert isinstance(ExperimentAgent(), Agent)


def test_base_methods():
    agent = UtilityAgent()
    assert agent.act("x") == "Performed x"
    assert agent.act_all(["a", "b"]) == ["Performed a", "Performed b"]


def test_experiment_agent_run():
    agent = ExperimentAgent()
    result = agent.run("hypothesis")
    assert result == "Experimenting with hypothesis"


def test_experiment_agent_run_series():
    agent = ExperimentAgent()
    results = agent.run_series(["h1", "h2"])
    assert results == ["Experimenting with h1", "Experimenting with h2"]
