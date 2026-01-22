import pickle
from pathlib import Path
from scrapy import signals
from scrapy.exceptions import NotConfigured
from scrapy.utils.job import job_dir
Store and load spider state during a scraping job