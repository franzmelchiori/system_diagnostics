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

import data_viewer


def get_pd_dataframe_minimum_sampling_period(pd_dataframe,
                                             sampling_precision='1s'):
    pd_dataframe_sampling_periods = []
    pd_second_delta = pd.to_timedelta(sampling_precision)
    if not pd_dataframe.empty:
        pd_dataframe_range = range(pd_dataframe.shape[0] - 1)
        for current_index in pd_dataframe_range:
            next_index = current_index + 1
            current_timestamp = pd_dataframe.index[current_index]
            next_timestamp = pd_dataframe.index[next_index]
            pd_dataframe_delta = current_timestamp - next_timestamp
            pd_dataframe_sampling_periods.append(int(
                pd_dataframe_delta/pd_second_delta))
    if pd_dataframe_sampling_periods:
        pd_dataframe_sampling_period = min(pd_dataframe_sampling_periods)
    else:
        pd_dataframe_sampling_period = False
    return pd_dataframe_sampling_period


def get_pd_dataframes_minimum_sampling_period(pd_dataframes,
                                              sampling_precision='1s'):
    pd_dataframes_sampling_periods = []
    for pd_dataframe in pd_dataframes:
        if not pd_dataframe.empty:
            pd_dataframe_sampling_period = \
                get_pd_dataframe_minimum_sampling_period(
                    pd_dataframe, sampling_precision)
            if pd_dataframe_sampling_period:
                pd_dataframes_sampling_periods.append(
                    pd_dataframe_sampling_period)
    if pd_dataframes_sampling_periods:
        pd_dataframes_minimum_sampling_period = min(
            pd_dataframes_sampling_periods)
    else:
        pd_dataframes_minimum_sampling_period = False
    return pd_dataframes_minimum_sampling_period


def get_down_rounded_sampling_period(raw_sampling_period, sampling_unit='s'):
    if sampling_unit == 's':
        hour_max_sampling_period = 3600
        down_rounded_sampling_period = 3600
    elif sampling_unit == 'ms':
        hour_max_sampling_period = 3600000
        down_rounded_sampling_period = 3600000
    else:
        hour_max_sampling_period = 3600
        down_rounded_sampling_period = 3600
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


def get_pd_dataframes_down_rounded_sampling_period(pd_dataframes,
                                                   sampling_precision='1s'):
    sampling_unit = get_sampling_unit(sampling_precision)
    pd_dataframes_down_rounded_sampling_period = False
    if pd_dataframes:
        pd_dataframes_minimum_sampling_period =\
            get_pd_dataframes_minimum_sampling_period(pd_dataframes,
                                                      sampling_precision)
        if pd_dataframes_minimum_sampling_period:
            pd_dataframes_down_rounded_sampling_period = \
                get_down_rounded_sampling_period(
                    pd_dataframes_minimum_sampling_period, sampling_unit)
    return pd_dataframes_down_rounded_sampling_period


def resample_pd_dataframes(pd_dataframes, sampling_precision='1s'):
    sampling_unit = get_sampling_unit(sampling_precision)
    if pd_dataframes:
        resampling_period = get_pd_dataframes_down_rounded_sampling_period(
            pd_dataframes, sampling_precision)
        resampling_period_string = '{0}{1}'.format(resampling_period,
                                                   sampling_unit)
        for pd_dataframe in pd_dataframes:
            if not pd_dataframe.empty:
                print(pd_dataframe)
                pd_dataframe = pd_dataframe.resample(
                    resampling_period_string).mean()
                print(pd_dataframe)
            break


def get_sampling_unit(sampling_precision):
    available_sampling_units = ['m', 's', 'ms']
    available_sampling_units_string = ''.join(available_sampling_units)
    sampling_precision_value = sampling_precision.strip(
        available_sampling_units_string)
    numbers_string = ''.join([str(n) for n in range(10)])
    sampling_unit = sampling_precision.strip(numbers_string)
    if not sampling_precision_value.isdecimal():
        sampling_unit = False
    if sampling_unit not in available_sampling_units:
        sampling_unit = False
    return sampling_unit


def main():
    # print(get_down_rounded_sampling_period(7))
    # print(get_sampling_unit('1321s'))
    pass


if __name__ == '__main__':
    main()