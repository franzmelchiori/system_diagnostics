#!/usr/bin/python3

"""
    System diagnostics: data sampler
    Copyright (C) 2019 Francesco Melchiori
    <https://www.francescomelchiori.com/>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import pandas as pd


def get_pd_dataframe_sampling_period(pd_dataframe):
    pd_dataframe_delta = pd_dataframe.index[0] - pd_dataframe.index[1]
    pd_second_delta = pd.to_timedelta('1s')
    pd_dataframe_sampling_period = int(pd_dataframe_delta/pd_second_delta)
    return pd_dataframe_sampling_period


def get_pd_dataframes_minimum_sampling_period(pd_dataframes):
    pd_dataframes_sampling_periods = []
    for pd_dataframe in pd_dataframes:
        pd_dataframe_sampling_period = get_pd_dataframe_sampling_period(
            pd_dataframe)
        pd_dataframes_sampling_periods.append(pd_dataframe_sampling_period)
    pd_dataframes_minimum_sampling_period = min(pd_dataframes_sampling_periods)
    print(pd_dataframes_sampling_periods)
    return pd_dataframes_minimum_sampling_period


def main():
    pass


if __name__ == '__main__':
    main()
