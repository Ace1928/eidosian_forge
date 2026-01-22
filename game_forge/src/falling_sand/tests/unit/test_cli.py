import pytest

from falling_sand import cli


def test_build_parser_has_commands() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["bench"])
    assert args.command == "bench"


def test_resolve_command_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown command"):
        cli.resolve_command("unknown")


def test_resolve_command_known() -> None:
    for name in ("demo", "bench", "index", "ingest", "report"):
        assert callable(cli.resolve_command(name))
