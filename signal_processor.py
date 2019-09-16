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
from scipy.fftpack import fft, fftfreq
from scipy import signal
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('seaborn-dark')


def signal_to_fit(x, a, b, c, d):
    return a * np.sin(b * x) + c * np.sin(d * x)


def main():
    sampling_period_s = 1
    sampling_frequency_s = 1/sampling_period_s
    processing_period_s = 60
    sampling_points = int(processing_period_s/sampling_period_s) + 1
    harmonic_base_period = (2 * np.pi) / processing_period_s
    harmonic_base_amplitude = 4
    harmonic_1_period = harmonic_base_period * 4
    harmonic_1_amplitude = 1
    noise_amplitude = 0.3
    interpolation_fraction = 5
    scatter_noisy = True
    scatter_interpolation_linear = True
    scatter_interpolation_cubic = True
    scatter_fitting = True

    sampling_times = np.linspace(0, processing_period_s, sampling_points)
    for sampling_time in sampling_times:
        plt.axvline(sampling_time, c='black', alpha=0.02)

    harmonic_base = np.sin(harmonic_base_period * sampling_times) * \
                    harmonic_base_amplitude
    harmonic_1 = np.sin(harmonic_1_period * sampling_times) * \
                 harmonic_1_amplitude
    noise = default_rng().standard_normal(size=sampling_points) * \
            noise_amplitude
    measures_clean = harmonic_base + harmonic_1
    measures_noisy = measures_clean + noise
    plt.scatter(sampling_times,
                measures_clean,
                marker='o', c='green', alpha=0.3)
    if noise_amplitude > 0:
        if scatter_noisy:
            plt.scatter(sampling_times,
                        measures_noisy,
                        marker='o', c='red')

    linear_interpolation = interp1d(sampling_times[::interpolation_fraction],
                                    measures_noisy[::interpolation_fraction])
    linear_results = linear_interpolation(sampling_times)
    if scatter_interpolation_linear:
        plt.scatter(sampling_times[::interpolation_fraction],
                    linear_results[::interpolation_fraction],
                    c='blue')
        plt.scatter(sampling_times,
                    linear_results,
                    s=10,
                    c='blue',
                    alpha=0.3)
        plt.plot(sampling_times, linear_results, c='blue', alpha=0.3)

    cubic_interpolation = interp1d(sampling_times[::interpolation_fraction],
                                   measures_noisy[::interpolation_fraction],
                                   kind='cubic')
    cubic_results = cubic_interpolation(sampling_times)
    if scatter_interpolation_cubic:
        plt.scatter(sampling_times[::interpolation_fraction],
                    cubic_results[::interpolation_fraction],
                    c='blue')
        plt.scatter(sampling_times,
                    cubic_results,
                    s=10,
                    c='blue')
        plt.plot(sampling_times, cubic_results, c='blue')

    params, params_covariance = curve_fit(
        signal_to_fit,
        sampling_times[::interpolation_fraction],
        measures_noisy[::interpolation_fraction],
        p0=[harmonic_base_amplitude, harmonic_base_period,
            harmonic_1_amplitude, harmonic_1_period])
    fitting_results = signal_to_fit(sampling_times, *params)
    if scatter_fitting:
        plt.scatter(sampling_times[::interpolation_fraction],
                    fitting_results[::interpolation_fraction],
                    c='purple')
        plt.scatter(sampling_times,
                    fitting_results,
                    s=10,
                    c='purple')
        plt.plot(sampling_times, fitting_results, c='purple')

    plt.show()

    measures_fft = fft(measures_noisy)
    measures_power = np.abs(measures_fft)
    measures_frequencies = fftfreq(measures_noisy.size,
                                   d=sampling_frequency_s)
    for measures_frequency in measures_frequencies:
        plt.axvline(measures_frequency, c='black', alpha=0.02)
    plt.scatter(measures_frequencies,
                measures_power,
                s=10,
                c='purple')

    plt.show()


if __name__ == '__main__':
    main()
