from scrapy.exceptions import NotConfigured, NotSupported
from scrapy.selector import Selector
from scrapy.spiders import Spider
from scrapy.utils.iterators import csviter, xmliter_lxml
from scrapy.utils.spider import iterate_spider_output
def parse_rows(self, response):
    """Receives a response and a dict (representing each row) with a key for
        each provided (or detected) header of the CSV file.  This spider also
        gives the opportunity to override adapt_response and
        process_results methods for pre and post-processing purposes.
        """
    for row in csviter(response, self.delimiter, self.headers, quotechar=self.quotechar):
        ret = iterate_spider_output(self.parse_row(response, row))
        for result_item in self.process_results(response, ret):
            yield result_item