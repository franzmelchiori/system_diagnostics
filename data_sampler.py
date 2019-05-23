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
    along with this program.  If not, see
    <http://www.gnu.org/licenses/>.
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
            pd_dataframe_delta = abs(current_timestamp - next_timestamp)
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


def pad_pd_dataframes(pd_dataframes, timestamp_start, timestamp_end,
                      time_zone):
    """
       For each series in each dataframe, pad_pd_dataframes pads an
       entry at the beginning and one at the end respectedly with the
       first and the last available values at the given timestamps.
    """
    pd_timezone_index_start = pd.DatetimeIndex([timestamp_start], tz=time_zone)
    pd_timezone_index_end = pd.DatetimeIndex([timestamp_end], tz=time_zone)
    pd_utc_index_start = pd.to_datetime(pd_timezone_index_start, utc=True)
    pd_utc_index_end = pd.to_datetime(pd_timezone_index_end, utc=True)
    pd_padded_dataframes = []
    if pd_dataframes:
        for pd_dataframe in pd_dataframes:
            if not pd_dataframe.empty:
                padding_start_dict = {}
                for pd_series in pd_dataframe.columns:
                    pd_series_values = pd_dataframe[pd_series]
                    for pd_series_value in pd_series_values:
                        if not pd.isnull(pd_series_value):
                            padding_start_dict['{0}'.format(pd_series)] = \
                                pd.Series(pd_series_value,
                                          index=pd_utc_index_start)
                            break
                padding_end_dict = {}
                for pd_series in pd_dataframe.columns:
                    pd_series_values = pd_dataframe[pd_series]
                    pd_series_size = pd_dataframe[pd_series].size
                    pd_series_back_index = [-pd_series_index
                                            for pd_series_index
                                            in range(1, pd_series_size+1)]
                    pd_series_back_values = pd_series_values[
                        pd_series_back_index]
                    for pd_series_back_value in pd_series_back_values:
                        if not pd.isnull(pd_series_back_value):
                            padding_end_dict['{0}'.format(pd_series)] = \
                                pd.Series(pd_series_back_value,
                                          index=pd_utc_index_end)
                            break
                pd_dataframe_padding_start = pd.DataFrame(padding_start_dict)
                pd_dataframe_padding_end = pd.DataFrame(padding_end_dict)
                pd_padded_dataframe = pd.concat([pd_dataframe,
                                                 pd_dataframe_padding_start,
                                                 pd_dataframe_padding_end])
                pd_padded_dataframes.append(pd_padded_dataframe)
    return pd_padded_dataframes


def resample_pd_dataframes(pd_dataframes, sampling_precision='1s'):
    """
       For each series in each dataframe, resample_pd_dataframes
       resamples the time series at the maximum sampling frequency among
       them, so upsampling the others.
    """
    sampling_unit = get_sampling_unit(sampling_precision)
    pd_resampled_dataframes = []
    if pd_dataframes:
        resampling_period = get_pd_dataframes_down_rounded_sampling_period(
            pd_dataframes, sampling_precision)
        resampling_period_string = '{0}{1}'.format(resampling_period,
                                                   sampling_unit)
        for pd_dataframe in pd_dataframes:
            if not pd_dataframe.empty:
                pd_resampled_dataframe = pd_dataframe.resample(
                    resampling_period_string).pad()
                pd_resampled_dataframes.append(pd_resampled_dataframe)
    return pd_resampled_dataframes


def fill_pd_dataframes(pd_dataframes):
    pd_filled_dataframes = []
    if pd_dataframes:
        for pd_dataframe in pd_dataframes:
            if not pd_dataframe.empty:
                pd_filled_dataframe = pd_dataframe.fillna(method='ffill')
                pd_filled_dataframes.append(pd_filled_dataframe)
    return pd_filled_dataframes


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

    timestamp_start = '2019-01-29 08:07:36.910000'
    timestamp_end = '2019-01-29 08:17:36.910000'
    resampling_timestamp_start = '2019-01-29 08:00:00'
    resampling_timestamp_end = '2019-01-29 09:00:00'
    time_zone = 'Europe/Rome'

    data_test_01 = np.random.normal(0, 1, 10)
    timezone_index_test_01 = pd.date_range(timestamp_start, periods=10,
                                           freq='5T', tz=time_zone)
    utc_index_test_01 = pd.to_datetime(timezone_index_test_01, utc=True)
    pd_series_test_01 = pd.Series(data_test_01, index=utc_index_test_01)
    data_test_02 = np.random.normal(0, 1, 6)
    timezone_index_test_02 = pd.date_range(timestamp_end, periods=6,
                                           freq='5T', tz=time_zone)
    utc_index_test_02 = pd.to_datetime(timezone_index_test_02, utc=True)
    pd_series_test_02 = pd.Series(data_test_02, index=utc_index_test_02)
    pd_dataframe_test_01 = pd.DataFrame(
        {'pd_series_test_01': pd_series_test_01,
         'pd_series_test_02': pd_series_test_02})

    data_test_03 = np.random.normal(0, 1, 10)
    pd_series_test_03 = pd.Series(data_test_03, index=utc_index_test_01)
    data_test_04 = np.random.normal(0, 1, 6)
    pd_series_test_04 = pd.Series(data_test_04, index=utc_index_test_02)
    pd_dataframe_test_02 = pd.DataFrame(
        {'pd_series_test_03': pd_series_test_03,
         'pd_series_test_04': pd_series_test_04})

    pd_dataframes = [pd_dataframe_test_01, pd_dataframe_test_02]

    for pd_dataframe in pd_dataframes:
        print(pd_dataframe)
    print()
    pd_dataframes = pad_pd_dataframes(pd_dataframes,
                                      resampling_timestamp_start,
                                      resampling_timestamp_end,
                                      time_zone)
    pd_dataframes = resample_pd_dataframes(pd_dataframes)
    pd_dataframes = fill_pd_dataframes(pd_dataframes)
    for pd_dataframe in pd_dataframes:
        print(pd_dataframe)
    data_viewer.view_pd_dataframes(pd_dataframes)


if __name__ == '__main__':
    main()
