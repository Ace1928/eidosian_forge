from urllib.error import URLError
import altair as alt
import pandas as pd
import streamlit as st
from streamlit.hello.utils import show_code
@st.cache_data
def get_UN_data():
    AWS_BUCKET_URL = 'https://streamlit-demo-data.s3-us-west-2.amazonaws.com'
    df = pd.read_csv(AWS_BUCKET_URL + '/agri.csv.gz')
    return df.set_index('Region')