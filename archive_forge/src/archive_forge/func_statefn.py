import pickle
from pathlib import Path
from scrapy import signals
from scrapy.exceptions import NotConfigured
from scrapy.utils.job import job_dir
@property
def statefn(self) -> str:
    return str(Path(self.jobdir, 'spider.state'))