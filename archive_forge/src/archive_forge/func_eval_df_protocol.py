import pandas
import modin.pandas as pd
from modin.pandas.utils import from_dataframe
from modin.tests.pandas.utils import df_equals, test_data
from modin.tests.test_utils import warns_that_defaulting_to_pandas
def eval_df_protocol(modin_df_producer):
    internal_modin_df_producer = modin_df_producer.__dataframe__()
    with warns_that_defaulting_to_pandas():
        modin_df_consumer = from_dataframe(modin_df_producer)
        internal_modin_df_consumer = from_dataframe(internal_modin_df_producer)
    assert modin_df_producer is not modin_df_consumer
    assert internal_modin_df_producer is not internal_modin_df_consumer
    assert modin_df_producer._query_compiler._modin_frame is not modin_df_consumer._query_compiler._modin_frame
    df_equals(modin_df_producer, modin_df_consumer)
    df_equals(modin_df_producer, internal_modin_df_consumer)