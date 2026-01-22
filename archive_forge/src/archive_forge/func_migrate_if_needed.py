from pathlib import Path
from alembic.command import upgrade
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy.engine.base import Engine
def migrate_if_needed(engine: Engine, revision: str) -> None:
    alembic_cfg = _get_alembic_config(engine.url.render_as_string(hide_password=False))
    script_dir = ScriptDirectory.from_config(alembic_cfg)
    with engine.begin() as conn:
        context = MigrationContext.configure(conn)
        if context.get_current_revision() != script_dir.get_current_head():
            upgrade(alembic_cfg, revision)