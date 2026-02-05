from __future__ import annotations

from ECosmos import config


def test_environment_factors_valid() -> None:
    assert config.ENVIRONMENT_FACTORS
    for factor in config.ENVIRONMENT_FACTORS:
        assert 0.0 <= factor.influence_strength <= 1.0
        assert factor.distribution in {"uniform", "gradient", "radial", "patches"}
        assert factor.name


def test_instruction_costs_are_positive() -> None:
    assert config.INSTRUCTION_COSTS
    for name, cost in config.INSTRUCTION_COSTS.items():
        assert isinstance(name, str)
        assert cost > 0.0


def test_visualization_colors_are_rgb() -> None:
    for mapping in config.COLOR_MAPPING.values():
        for entry in mapping.values():
            assert len(entry) == 3
            assert all(0 <= value <= 255 for value in entry)
