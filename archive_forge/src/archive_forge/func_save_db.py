from ._base import *
from .types import CreateSchemaType, UpdateSchemaType
from .models import LazyHasher, LazyUserSchema, LazyDBConfig, LazyDBSaveMetrics
def save_db(self):
    self.metrics.last_save = tstamp()
    self.metrics.num_saved += 1
    self.metrics.time_alive = self._alivetime.ablstime
    dbdata = {'db': self._db, 'timer': self._alivetime, 'metrics': self.metrics}
    self.cache.save(dbdata)