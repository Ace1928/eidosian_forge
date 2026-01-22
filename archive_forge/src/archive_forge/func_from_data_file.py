from urllib.error import URLError
import pandas as pd
import pydeck as pdk
import streamlit as st
from streamlit.hello.utils import show_code
@st.cache_data
def from_data_file(filename):
    url = 'https://raw.githubusercontent.com/streamlit/example-data/master/hello/v1/%s' % filename
    return pd.read_json(url)