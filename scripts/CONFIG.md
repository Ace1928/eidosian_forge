# Configuration

Each script can read an optional env file from `~/.config/eidosian/<script>.env`.

Example:
```
# ~/.config/eidosian/overwrite.env
OVERWRITE_DEFAULT_FILE=file_tree.txt
OVERWRITE_BACKUP_DIR=~/.local/share/eidos_backups
```

Environment variables take precedence over defaults, then CLI flags override env.

## Treegen config file
`treegen` supports `--config /path/to/file` to source a shell config file containing
`EIDOSIAN_` variables (for example `EIDOSIAN_use_color=false`).
