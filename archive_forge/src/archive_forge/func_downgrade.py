from debtcollector import removals
import sqlalchemy
from stevedore import enabled
from oslo_db import exception
def downgrade(self, revision):
    """Downgrade database with available backends."""
    rev_in_plugins = [p.has_revision(revision) for p in self._plugins]
    if not any(rev_in_plugins) and revision is not None:
        raise exception.DBMigrationError('Revision does not exist')
    results = []
    for plugin, has_revision in zip(reversed(self._plugins), reversed(rev_in_plugins)):
        if not has_revision or revision is None:
            results.append(plugin.downgrade(None))
        else:
            results.append(plugin.downgrade(revision))
            break
    return results