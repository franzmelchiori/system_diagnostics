#!/usr/bin/python3

"""
    System diagnostics: data viewer
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

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


def view_pd_dataframe(pd_dataframe):
    pd_dataframe_keys = pd_dataframe.keys()
    pd_dataframe_keys_size = pd_dataframe_keys.size
    if pd_dataframe_keys_size == 1:
        plt.figure()
        pd_dataframe_label = pd_dataframe.keys()[0]
        plt.plot(pd_dataframe, label=pd_dataframe_label)
        plt.legend()
    elif pd_dataframe_keys_size >= 2:
        pd_dataframe_zip = zip(range(pd_dataframe_keys_size), pd_dataframe_keys)
        fig, ax = plt.subplots(pd_dataframe_keys_size)
        for (key_count, pd_dataframe_key) in pd_dataframe_zip:
            pd_dataframe_label = pd_dataframe.keys()[key_count]
            ax[key_count].plot(pd_dataframe[pd_dataframe_key])
            ax[key_count].set(label=pd_dataframe_label)
            ax[key_count].legend()
    plt.show()
    return True


def view_pd_dataframes(pd_dataframes):
    for pd_dataframe in pd_dataframes:
        view_pd_dataframe(pd_dataframe)
    return True


def main():
    pass


if __name__ == '__main__':
    main()
