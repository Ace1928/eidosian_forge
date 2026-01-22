import pandas as pd
import matplotlib.pyplot as plt
import sys
def compare_styles(s1, s2):
    with plt.rc_context():
        plt.style.use('default')
        plt.style.use(s1)
        df1 = rcParams_to_df(plt.rcParams, name=s1)
    with plt.rc_context():
        plt.style.use('default')
        plt.style.use(s2)
        df2 = rcParams_to_df(plt.rcParams, name=s2)
    df = pd.concat([df1, df2], axis=1)
    dif = df[df[s1] != df[s2]].dropna(how='all')
    return (dif, df, df1, df2)