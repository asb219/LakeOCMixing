"""
`mcmc.py`: Markov Chain Monte Carlo calibration of an isotopic mixing model
for organic carbon in lake sediments.

Copyright (C) 2024  Alexander S. Brunmayr and Benedict V. A. Mittelbach

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import pathlib
from datetime import datetime

import pandas as pd
import numpy as np
from numba import njit
import matplotlib.pyplot as plt

import pymc as pm
import pytensor.tensor as pt
import arviz as az
from corner import corner


USE_SYNTHETIC_DATA = False # use synthetic_data instead of observed_data

PLOTS_FILE_EXTENSION = '.svg' # '.pdf'

NUMBER_OF_CORES = 8 # max number of CPU cores to use for MCMC

RANDOM_NUMBER_GENERATOR_SEED = 0 # for exact reproducibility
RANDOM_NUMBER_GENERATOR = np.random.default_rng(RANDOM_NUMBER_GENERATOR_SEED)

DECAY_RATE_14C = 1.21e-4 # radioactive decay rate of 14C (per year)

REPODIR = pathlib.Path(__file__).resolve().parent # directory of the repository


#### READ OBSERVED DATA ####

observed_data = pd.read_excel(
    REPODIR / 'input' / 'sedimentcore_data.xlsx', sheet_name='input_data'
).rename(columns={'Year of deposition': 'year'}).set_index('year').sort_index()

# Exclude flood layers and layers with incomplete data
observed_data = observed_data[~observed_data['Flood']]
observed_data = observed_data.dropna(subset=['D14C','d13C','ROC_TOC','TOC_TN'])

# Average over duplicate measurements
observed_data = observed_data.drop(columns='ETH ID').groupby('year').mean()

# Transform D14C into age-corrected absolute F14C
observed_years = observed_data.index
sampling_year = 2022
observed_data['F14C'] = (observed_data['D14C']/1000 + 1) \
    * np.exp(DECAY_RATE_14C * (sampling_year - observed_years))

# Measurement uncertainties
observed_data['F14C_sigma'] = 0.008
observed_data['d13C_sigma'] = 0.15
observed_data['ROC_TOC_sigma'] = 0.02
observed_data['TOC_TN_sigma'] = 0.5



#### Set up 1-pool model for 14C simulations ####

F14C_atm = pd.read_csv(REPODIR / 'input' / 'Delta14CO2_atmosphere.csv',
    index_col='year', usecols=['year', 'Delta14C']).squeeze() / 1000 + 1


@njit
def F14C_1pool_model(k, out_years, influx_years, influx_F14C):
    # Note: out_years must be sorted

    count = 0
    max_count = len(out_years)
    F14C_array = np.empty(max_count, dtype=np.float64)

    k_lambda14C = k + DECAY_RATE_14C
    F14C = k / k_lambda14C # steady-state pre-industrial F14C with F14C_in = 1

    for year, F14C_in in zip(influx_years, influx_F14C):

        F14C += (k * F14C_in) - (k_lambda14C * F14C)

        while out_years[count] == year:
            F14C_array[count] = F14C
            count += 1
            if count == max_count:
                return F14C_array

    raise ValueError


class ODEop(pt.Op):

    itypes = [pt.dscalar]
    otypes = [pt.dvector]

    def __init__(self, out_years, lag):
        start_year = 1880
        end_year = 2020
        influx_years = F14C_atm.loc[start_year-1 : end_year].index.values
        influx_F14C = F14C_atm.loc[start_year-1-lag : end_year-lag].values
        self._function = \
            lambda k: F14C_1pool_model(k, out_years, influx_years, influx_F14C)

    def perform(self, node, inputs_storage, output_storage):
        x = inputs_storage[0]
        out = output_storage[0]
        out[0] = self._function(x)



if USE_SYNTHETIC_DATA:

    #### PRODUCE SYNTHETIC DATA ####

    years = observed_years.values
    Nyears = len(years)

    rng = RANDOM_NUMBER_GENERATOR

    source_contributions = pd.DataFrame(
        [[0.3, 0.2, 0.5]]*Nyears,
        columns=['aq','petr','soil'], index=years
    )
    source_contributions += rng.normal(0, 0.05, size=(Nyears,3))
    source_contributions /= source_contributions.values.sum(axis=1)[:,None]

    source_d13C = pd.DataFrame(
        [[-32.5, -22.5, -27]]*Nyears, #index=['aq','petr','soil']
        columns=['aq','petr','soil'], index=years
    ) + np.array([
        [1.5 if 1955 < y < 1985 else 0, 0, 0] for y in years
    ], dtype=np.float64) # autochthonous d13C is higher during eutrophication
    source_d13C += rng.normal(0, 0.5, size=(Nyears,3))

    turnover_times = pd.Series([5., 400.], index=['aq','soil'])

    source_CN = pd.DataFrame(
        [[6.96, 4.60, 12.33]]*Nyears, #index=['aq','petr','soil']
        columns=['aq','petr','soil'], index=years
    )
    source_CN += rng.normal(0, 1, size=(Nyears,3))

    start_year = 1880
    end_year = 2020

    k = 1. / turnover_times['aq']
    lag = 1
    influx_years = F14C_atm.loc[start_year-1 : end_year].index.values
    influx_F14C = F14C_atm.loc[start_year-1-lag : end_year-lag].values
    F14C_aq = 0.8 * F14C_1pool_model(k, years, influx_years, influx_F14C)

    k = 1. / turnover_times['soil']
    lag = 3
    influx_years = F14C_atm.loc[start_year-1 : end_year].index.values
    influx_F14C = F14C_atm.loc[start_year-1-lag : end_year-lag].values
    F14C_soil = F14C_1pool_model(k, years, influx_years, influx_F14C)

    F14C = source_contributions['aq']*F14C_aq + source_contributions['soil']*F14C_soil
    d13C = (source_contributions * source_d13C).sum(axis=1)
    ROC_TOC = source_contributions['petr']
    TOC_TN = 1. / (source_contributions / source_CN).sum(axis=1)

    synthetic_data = pd.DataFrame({
        'F14C': F14C, 'F14C_sigma': 0.008, # 8 permille
        'd13C': d13C, 'd13C_sigma': 0.15, # 0.15 permille
        'ROC_TOC': ROC_TOC, 'ROC_TOC_sigma': 0.02,
        'TOC_TN': TOC_TN, 'TOC_TN_sigma': 0.5
    }, index=years)

    data = synthetic_data

else:
    data = observed_data



with pm.Model() as pymc_model:

    endmembers = ['autochthonous OC', 'petrogenic OC', 'soil OC']
    years = data.index.values
    shape = (len(years), len(endmembers))


    #### PRIOR DISTRIBUTIONS ####

    # Source contributions
    contrib = pm.Dirichlet('c', a=[1,1,1], shape=shape)

    # Source d13C
    d13C_mu_additional_modifier = np.array([
        [1.5 if 1955 < y < 1985 else 0, 0, 0] for y in years
    ], dtype=np.float64) # autochthonous d13C is higher during eutrophication
    d13C_sigma_multiplicative_modifier = np.array([
        [2 if 1955 < y < 1985 else 1, 1, 1] for y in years
    ], dtype=np.float64) # double the uncertainty during eutrophication period
    d13C_mu = np.array([-32.5, -22.5, -27]) + d13C_mu_additional_modifier
    d13C_sigma = np.array([0.2, 0.2, 0.2]) * d13C_sigma_multiplicative_modifier
    d13C = pm.Normal('d13C', mu=d13C_mu, sigma=d13C_sigma, shape=shape)

    # Source C/N ratios
    CN_mu = [6.96, 4.60, 12.33]
    CN_sigma = [0.3, 0.3, 0.3]
    CN_dist = pm.Normal.dist(mu=CN_mu, sigma=CN_sigma)
    CN = pm.Truncated('CN', CN_dist, lower=0., shape=shape) # ensure C/N > 0

    # F14C of autochthonous OC endmember
    tau_aq = pm.Uniform('tau_aq', lower=2, upper=30) # turnover time
    k_aq = pm.Deterministic('k_aq', 1. / tau_aq) # turnover rate
    hardwater_effect = 0.20 # 20% of lake DIC is rock-derived (with F14C = 0)
    lag_aq = 1 # lag time (years) for the C transfer from atmosphere to lake
    F14C_aq_Op = ODEop(years, lag_aq)
    F14C_aq = (1 - hardwater_effect) * F14C_aq_Op(k_aq)

    # F14C of soil OC endmember
    tau_soil = pm.Uniform('tau_soil', lower=10, upper=6000) # turnover time
    k_soil = pm.Deterministic('k_soil', 1. / tau_soil) # turnover rate
    lag_soil = 3 # lag time (years) for the C transfer from atmosphere to soil
    F14C_soil_Op = ODEop(years, lag=lag_soil)
    F14C_soil = F14C_aq_Op(k_soil)


    #### LIKELIHOOD ####

    # Sediment d13C
    d13C_sediment = (contrib * d13C).sum(axis=1)
    d13C_obs = pm.Normal('d13C_obs', mu=d13C_sediment,
        sigma=data['d13C_sigma'].values,
        observed=data['d13C'].values)

    # Sediment TOC/TN
    CN_sediment = 1. / (contrib / CN).sum(axis=1)
    CN_obs = pm.Normal('CN_obs', mu=CN_sediment,
        sigma=data['TOC_TN_sigma'].values,
        observed=data['TOC_TN'].values)

    # Sediment ROC/TOC
    ROC_TOC_sediment = contrib[:,1]
    ROC_TOC_obs = pm.Normal('ROC_TOC_obs', mu=ROC_TOC_sediment,
        sigma=data['ROC_TOC_sigma'].values,
        observed=data['ROC_TOC'].values)

    # Sediment F14C
    F14C_sediment = F14C_aq * contrib[:,0] + F14C_soil * contrib[:,2]
    F14C_obs = pm.Normal('F14C_obs', mu=F14C_sediment,
        sigma=data['F14C_sigma'].values,
        observed=data['F14C'].values)



if __name__=='__main__':

    # Sample the posterior distribution with the Metropolis-Hastings algorithm
    with pymc_model:
        step = pm.Metropolis() #(target_accept=0.9)
        draws = 50000
        tune = 2000
        chains = 6
        idata = pm.sample(
            step=step, draws=draws, tune=tune, chains=chains,
            cores=NUMBER_OF_CORES, random_seed=RANDOM_NUMBER_GENERATOR,
            compute_convergence_checks=False
        )

    # Create output directory
    OUTDIR = REPODIR / 'output' / 'mcmc' / ' '.join([
        'mcmc', 'synthetic' if USE_SYNTHETIC_DATA else 'observed', 'data',
        datetime.now().strftime('%Y-%m-%d %H.%M.%S')
    ])
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Write MCMC inference data to file
    idata.to_netcdf(OUTDIR / f'idata.nc')
    # To read this file, use az.from_netcdf

    # Produce summary of MCMC trace and write to CSV and Excel files
    summary = az.summary(idata)
    summary.to_csv(OUTDIR / f'summary.csv')
    summary.to_excel(OUTDIR / f'summary.xlsx')


    # Produce plots

    az.plot_trace(idata, var_names=['tau_aq','tau_soil'])
    plt.tight_layout()
    plt.savefig(OUTDIR / ('trace_turnovers' + PLOTS_FILE_EXTENSION))
    plt.close()

    if USE_SYNTHETIC_DATA:
        truths = [turnover_times['aq'], turnover_times['soil']]
    else:
        truths = None
    corner(idata, var_names=['tau_aq','tau_soil'], truths=truths)
    plt.savefig(OUTDIR / ('corner_turnovers' + PLOTS_FILE_EXTENSION))
    plt.close()

    idx = summary.index.str
    fig = plt.figure(figsize=(8,3))
    for i, endmember in enumerate(['aq', 'petr', 'soil']):
        ci = summary[idx.startswith('c[') & idx.endswith(f'{i}]')]
        mean = ci['mean']
        hdi3 = ci['hdi_3%']
        hdi97 = ci['hdi_97%']
        plt.fill_between(years, hdi3, hdi97, color=f'C{i}', alpha=0.2)
        plt.plot(years, mean, 'o-', color=f'C{i}', label=endmember)
    plt.ylim((0, 1))
    plt.ylabel('contribution to sediment OC')
    plt.xlabel('year')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / ('contributions_timeseries' + PLOTS_FILE_EXTENSION))
    plt.close()
