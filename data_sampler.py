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

import numpy as np
import pandas as pd


def get_pd_dataframe_sampling_period(pd_dataframe, sampling_precision='1s'):
    pd_dataframe_delta = pd_dataframe.index[0] - pd_dataframe.index[1]
    pd_second_delta = pd.to_timedelta(sampling_precision)
    pd_dataframe_sampling_period = int(pd_dataframe_delta/pd_second_delta)
    return pd_dataframe_sampling_period


def get_pd_dataframes_minimum_sampling_period(pd_dataframes,
                                              sampling_precision='1s'):
    pd_dataframes_sampling_periods = []
    for pd_dataframe in pd_dataframes:
        if not pd_dataframe.empty:
            pd_dataframe_sampling_period = get_pd_dataframe_sampling_period(
                pd_dataframe, sampling_precision)
        pd_dataframes_sampling_periods.append(pd_dataframe_sampling_period)
    pd_dataframes_minimum_sampling_period = min(pd_dataframes_sampling_periods)
    return pd_dataframes_minimum_sampling_period


def get_down_rounded_sampling_period(raw_sampling_period, sampling_unit='s'):
    if sampling_unit == 's':
        hour_max_sampling_period = 3600
    elif sampling_unit == 'ms':
        hour_max_sampling_period = 3600000
    hour_all_sampling_periods = np.arange(1, hour_max_sampling_period+1)
    hour_label_valid_sampling_periods = np.mod(hour_max_sampling_period,
                                               hour_all_sampling_periods) == 0
    hour_valid_sampling_periods = hour_all_sampling_periods[
        hour_label_valid_sampling_periods]
    hour_label_down_rounded_sampling_periods = np.floor_divide(
        hour_valid_sampling_periods, raw_sampling_period+1) == 0
    for label_sampling_period, sampling_period in zip(
            hour_label_down_rounded_sampling_periods,
            hour_valid_sampling_periods):
        if label_sampling_period:
            down_rounded_sampling_period = int(sampling_period)
        else:
            break
    return down_rounded_sampling_period


def main():
    print(get_down_rounded_sampling_period(7))


if __name__ == '__main__':
    main()
