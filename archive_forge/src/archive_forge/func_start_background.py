from ._base import *
from .types import CreateSchemaType, UpdateSchemaType
from .models import LazyHasher, LazyUserSchema, LazyDBConfig, LazyDBSaveMetrics
def start_background(self):
    self.env.enable_watcher()
    self.t = threading.Thread(target=self.background, daemon=True)
    self.t.start()
    self.env.add_thread(self.t)