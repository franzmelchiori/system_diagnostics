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
    along with this program.  If not, see
    <http://www.gnu.org/licenses/>.
"""


import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from sklearn.decomposition import PCA

plt.style.use('seaborn-dark')
register_matplotlib_converters()


def view_pd_dataframe(pd_dataframe, legend=True):
    pd_dataframe_keys = pd_dataframe.keys()
    pd_dataframe_keys_size = pd_dataframe_keys.size
    if pd_dataframe_keys_size == 1:
        plt.figure()
        pd_dataframe_label = pd_dataframe.keys()[0]
        plt.plot(pd_dataframe, label=pd_dataframe_label)
        plt.legend()
    elif pd_dataframe_keys_size >= 2:
        pd_dataframe_zip = zip(range(pd_dataframe_keys_size),
                               pd_dataframe_keys)
        fig, ax = plt.subplots(pd_dataframe_keys_size)
        for (key_count, pd_dataframe_key) in pd_dataframe_zip:
            pd_dataframe_label = pd_dataframe.keys()[key_count]
            ax[key_count].plot(pd_dataframe[pd_dataframe_key])
            ax[key_count].set(label=pd_dataframe_label)
            if legend:
                ax[key_count].legend()
    plt.show()
    return True


def view_pd_dataframes(pd_dataframes):
    for pd_dataframe in pd_dataframes:
        view_pd_dataframe(pd_dataframe)
    return True


def scatter_pd_series_2d(pd_series, pd_series_cluster_labels=None,
                         pd_series_cluster_centers=None,
                         pd_series_closest_cluster_center_labels=None):
    pca = PCA(n_components=2)
    pca.fit(pd_series)
    pd_series_pca = pca.transform(pd_series)
    pd_series_2d = pca.inverse_transform(pd_series_pca)
    point_groups = pd_series_cluster_labels
    if pd_series_cluster_labels is not None:
        point_groups = pd_series_cluster_labels
    plt.scatter(pd_series_2d[:, 0], pd_series_2d[:, 1], c=point_groups)
    if pd_series_cluster_centers is not None:
        plt.scatter(pd_series_cluster_centers[:, 0],
                    pd_series_cluster_centers[:, 1],
                    marker='x', color='red')
    if pd_series_closest_cluster_center_labels is not None:
        for pd_series_closest_cluster_center_label in \
                pd_series_closest_cluster_center_labels:
            plt.scatter(
                pd_series_2d[int(pd_series_closest_cluster_center_label), 0],
                pd_series_2d[int(pd_series_closest_cluster_center_label), 1])
    plt.show()


def main():
    pass


if __name__ == '__main__':
    main()
