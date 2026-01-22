from .roundtrip import round_trip  # , dedent, round_trip_load, round_trip_dump
def test_so_39595807(self):
    inp = "        %YAML 1.2\n        ---\n        [2, 3, 4]:\n          a:\n          - 1\n          - 2\n          b: Hello World!\n          c: 'Voilà!'\n        "
    round_trip(inp, preserve_quotes=True, explicit_start=True, version=(1, 2))