#!/usr/bin/python3

"""
    System diagnostics: signal processor
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
from numpy.random import default_rng
import scipy as sp
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
# scipy.fftpack
# scipy.signal
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('seaborn-dark')


def signal_to_fit(x, a, b):
    return a * np.sin(b * x)
# def signal_to_fit(x, a, b, c, d):
#     return a * np.sin(2 * np.pi * b * x) + c * np.sin(2 * np.pi * d * x)


def main():
    sampling_period_s = 1
    processing_period_s = 60
    sampling_points = int(processing_period_s/sampling_period_s) + 1
    harmonic_base_period = (2 * np.pi) / processing_period_s
    harmonic_base_amplitude = 4
    harmonic_1_period = harmonic_base_period * 4
    harmonic_1_amplitude = 1
    noise_amplitude = 0.3
    interpolation_fraction = 3
    scatter_noisy = True
    scatter_interpolation_linear = False
    scatter_interpolation_cubic = False
    scatter_fitting = True

    measurement_time = np.linspace(0, processing_period_s, sampling_points)
    plt.scatter(measurement_time,
                np.zeros(measurement_time.size),
                marker='x', c='black', alpha=0.1)

    harmonic_base = np.sin(harmonic_base_period * measurement_time) * \
                    harmonic_base_amplitude
    harmonic_1 = np.sin(harmonic_1_period * measurement_time) * \
                 harmonic_1_amplitude
    noise = default_rng().standard_normal(size=sampling_points) * \
            noise_amplitude
    measures_clean = harmonic_base + harmonic_1
    measures_noisy = measures_clean + noise
    plt.scatter(measurement_time,
                measures_clean,
                marker='o', c='green', alpha=0.3)
    if noise_amplitude > 0:
        if scatter_noisy:
            plt.scatter(measurement_time,
                        measures_noisy,
                        marker='o', c='red')

    linear_interpolation = interp1d(measurement_time[::interpolation_fraction],
                                    measures_noisy[::interpolation_fraction])
    linear_results = linear_interpolation(measurement_time)
    if scatter_interpolation_linear:
        plt.scatter(measurement_time[::interpolation_fraction],
                    linear_results[::interpolation_fraction],
                    c='blue')
        plt.scatter(measurement_time,
                    linear_results,
                    s=10,
                    c='blue',
                    alpha=0.3)
        plt.plot(measurement_time, linear_results, c='blue', alpha=0.3)

    cubic_interpolation = interp1d(measurement_time[::interpolation_fraction],
                                   measures_noisy[::interpolation_fraction],
                                   kind='cubic')
    cubic_results = cubic_interpolation(measurement_time)
    if scatter_interpolation_cubic:
        plt.scatter(measurement_time[::interpolation_fraction],
                    cubic_results[::interpolation_fraction],
                    c='blue')
        plt.scatter(measurement_time,
                    cubic_results,
                    s=10,
                    c='blue')
        plt.plot(measurement_time, cubic_results, c='blue')

    params, params_covariance = curve_fit(
        signal_to_fit, measurement_time, measures_noisy,
        p0=[harmonic_base_amplitude, harmonic_base_period])
    fitting_results = signal_to_fit(measurement_time, *params)
    if scatter_fitting:
        plt.scatter(measurement_time,
                    fitting_results,
                    c='blue')

    plt.show()


if __name__ == '__main__':
    main()
