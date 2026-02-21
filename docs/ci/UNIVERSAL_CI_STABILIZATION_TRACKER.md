# Universal CI Stabilization Tracker

## Cycle 2026-02-21

- [x] ~~Collect latest failing `Eidosian Universal CI` run diagnostics with `gh run view` and isolate failing jobs.~~
- [x] ~~Identify concrete failing test groups from CI logs (`game_forge`, `glyph_forge`).~~
- [x] ~~Harden `game_forge` scipy-fallback KDTree stub to support `query_ball_point` in no-scipy CI environments.~~
- [x] ~~Add regression assertion for `query_ball_point` fallback behavior in `game_forge/tests/test_gene_particles_types.py`.~~
- [x] ~~Gate streaming-specific glyph tests behind `pytest.importorskip("cv2")` to keep optional dependency behavior explicit.~~
- [x] ~~Run full local universal test sweep across all discovered forge test directories with CI-style flags.~~
- [x] ~~Verify no failing forge test groups remain in local universal sweep.~~
- [x] ~~Run and verify latest GitHub `Eidosian Universal CI` for commit `30194b31f0`; captured failing run: `https://github.com/Ace1928/eidosian_forge/actions/runs/22255932094` (`glyph_forge` renderer quality tests required cv2).~~
- [x] ~~Patch remaining glyph quality tests to skip cleanly when OpenCV is unavailable (`glyph_forge/tests/test_renderer_quality.py`).~~
- [x] ~~Patch `glyph_forge/tests/test_streaming_engine_targets.py` to skip when OpenCV is unavailable (`a5dc159924`).~~
- [x] ~~Harden optional dependency stubs in `glyph_forge/src/glyph_forge/streaming/extractors.py` and stabilize retry timeout sensitivity in `agent_forge/tests/test_retries.py` (`d1e2795517`).~~
- [x] ~~Re-run `Eidosian Universal CI` after `d1e2795517` and capture failure run `https://github.com/Ace1928/eidosian_forge/actions/runs/22256896857` (remaining failure: `glyph_forge/tests/test_streaming_extractors.py::test_extract_playlist_stitches_audio` requiring host `ffmpeg`).~~
- [x] ~~Patch playlist stitching test to mock `ffmpeg` binary discovery with `shutil.which` so the test remains environment-independent (`glyph_forge/tests/test_streaming_extractors.py`).~~

## Next Cycle Queue

- [x] ~~Re-run `Eidosian Universal CI` after the `ffmpeg`-discovery test patch and capture the final pass/fail status + run URL (`https://github.com/Ace1928/eidosian_forge/actions/runs/22257241042`, result: success).~~
- [x] ~~Triage non-universal workflow status (`workflow-lint`, `security-audit`, `secret-scan`): latest runs are green (`workflow-lint` `22226763756`, `security-audit` `22250091775`, `secret-scan` `22257866159`).~~
- [x] ~~Normalize `pytest_asyncio` loop scope config in repo test configuration to remove recurring warning noise in CI logs (set `asyncio_default_fixture_loop_scope=function` in `pytest.ini` and `agent_forge/pytest.ini`).~~
