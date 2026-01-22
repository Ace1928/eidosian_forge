from __future__ import annotations
import logging # isort:skip
from ..util.sampledata import external_csv

population_url = "http://spreadsheets.google.com/pub?key=phAwcNAVuyj0XOoBL_n5tAQ&output=xls"
fertility_url = "http://spreadsheets.google.com/pub?key=phAwcNAVuyj0TAlJeCEzcGQ&output=xls"
life_expectancy_url = "http://spreadsheets.google.com/pub?key=tiAiXcrneZrUnnJ9dBU-PAw&output=xls"
regions_url = "https://docs.google.com/spreadsheets/d/1OxmGUNWeADbPJkQxVPupSOK5MbAECdqThnvyPrwG5Os/pub?gid=1&output=xls"

def _get_data(url):
    # Get the data from the url and return only 1962 - 2013
    df = pd.read_excel(url, index_col=0)
    df = df.unstack().unstack()
    df = df[(df.index >= 1964) & (df.index <= 2013)]
    df = df.unstack().unstack()
    return df

fertility_df = _get_data(fertility_url)
life_expectancy_df = _get_data(life_expectancy_url)
population_df = _get_data(population_url)
regions_df = pd.read_excel(regions_url, index_col=0)

# have common countries across all data
fertility_df = fertility_df.drop(fertility_df.index.difference(life_expectancy_df.index))
population_df = population_df.drop(population_df.index.difference(life_expectancy_df.index))
regions_df = regions_df.drop(regions_df.index.difference(life_expectancy_df.index))

fertility_df.to_csv('gapminder_fertility.csv')
population_df.to_csv('gapminder_population.csv')
life_expectancy_df.to_csv('gapminder_life_expectancy.csv')
regions_df.to_csv('gapminder_regions.csv')
