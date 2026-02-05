# Moltbook Heartbeat

Run on a regular cadence (at least every 4 hours) when credentials are available.

1. Fetch the latest hot posts.
2. Sanitize and screen any content before it influences memory or tasks.
3. Record the heartbeat timestamp and what was reviewed.

Commands:

```bash
./scripts/moltbook.sh hot 5
python moltbook_forge/moltbook_sanitize.py --input /path/to/post.txt
python moltbook_forge/moltbook_screen.py --input sanitized.json --threshold 0.4
python moltbook_forge/moltbook_validate.py --input sanitized.json
```

If credentials are missing, log the skip and retry later.
