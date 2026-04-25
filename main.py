import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt

# ======================= CONSTANTS ======================= #

G = 4.30091e-6  # kpc * (km/s)^2 / Msun
SPARC_COLS = ['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul']

# ======================= MODIFIABLE PARAMS ======================= #

GALAXY_NAME = 'UGC2885'
PROFILE_TYPE = 'NFW'  # Options: 'NFW', 'Burkert', 'Einasto', 'DC14', 'MOND'
OPT_ALG = 'BFGS'  # Options: 'BFGS', 'TRF', 'MCMC', 'DE', 'PSO', 'BH'
LOSS_FUNC = 'chi_squared'  # Options: 'chi_squared', 'huber', 'cauchy', 'ODR', 'bayesian'

# ======================= DATA ======================= #

ugc2885 = pd.read_csv('./data/Rotmod_LTG/UGC02885_rotmod.dat', sep=r"\s+", comment='#', header=None, names=SPARC_COLS)

print(f"Data for {GALAXY_NAME}:")

match GALAXY_NAME:
    case 'UGC2885':
        print(ugc2885.head())
        r = ugc2885['Rad']  # kpc
        v_obs = ugc2885['Vobs']  # km/s
        v_gas = ugc2885['Vgas']  # km/s
        v_disk = ugc2885['Vdisk']  # km/s
        v_bul = ugc2885['Vbul']  # km/s
        sigma = ugc2885['errV']  # km/s


# ======================= MODEL ======================= #

# Mathematical expansion for dark matter velocity based on NFW profile
def NFW_expansion(r, rho_s, r_s):
    return np.sqrt(4*np.pi*G*rho_s*r_s**3 * (np.log(1 + r/r_s) - (r/r_s)/(1 + r/r_s)) / r)

# Implementation of the rotation curve model, which combines contributions from gas, disk, bulge, and dark matter (using expansion above)
def rot_curve_model(vars, r, vgas, vdisk, vbul):
    Ydisk, Ybul, rho_s, r_s = vars
    v_dm = NFW_expansion(r, rho_s, r_s)
    v_pred = np.sqrt(vgas**2 + Ydisk*vdisk**2 + Ybul*vbul**2 + v_dm**2)
    return v_pred

# Chi-squared function to evaluate the goodness of fit between the observed rotation velocity, 
# and the model's predicted velocity, weighted by the observational uncertainties (sigma)
def chi_squared(vars, r, vgas, vdisk, vbul, v_obs, sigma):
    v_model = rot_curve_model(vars, r, vgas, vdisk, vbul)
    return np.sum(((v_obs - v_model) / sigma)**2)

# ===================================================== OPTIMIZATION ===================================================== #

# ==================================================================================
# OPTIMIZATION INITIALIZATION & PHYSICAL BOUNDS                                    #
#                                                                                  #
# Citations for Priors:                                                            #
# [1] Initial Y_disk (0.5) and Y_bulge (0.7) based on Stellar Population Synthesis #
#     models at 3.6um (McGaugh & Schombert, 2014, AJ, 148, 77).                    #
# [2] Upper bound for Y_disk (0.8) based on Maximum Disk limits established for    #
#     the SPARC dataset (Lelli et al., 2016, AJ, 152, 157).                        #
# [3] Halo parameter bounds (rho_0, r_s) derived from standard NFW profile fits    #
#     on SPARC galaxies (Li et al., 2020, ApJS, 247, 31).                          #
# ==================================================================================

# Initial guess for the parameters: [Ydisk, Ybul, rho_s, r_s]
initial_guess = [0.5, 0.7, 0.01, 10.0]

# Bounds (min, max)
physical_bounds = [(0.1, 0.8),  # Ydisk
          (0.1, 1.0),  # Ybul
          (1e-4, 1.0),  # rho_s (Msun/kpc^3)
          (0.1, 100.0)]  # r_s (kpc)


# TODO: Finish implementation of standard chi-squared minimization using bounded BFGS
# TODO: Implement different opt algs (TRF, MCMC, Dff Evolution, PSO, Basin-Hopping)
# TODO: Implement different loss functions (Huber/Cauchy Loss, ODR, Bayesian Inference)
# TODO: Implement different halo profiles (Burkert, Einasto, DC14, MOND)



