import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt
from scipy.optimize import minimize
from scipy.special import gamma, gammainc
from scipy.integrate import quad
import warnings
from scipy.integrate import IntegrationWarning
# Suppress harmless SciPy integration warnings caused by the optimizer's gradient checks
warnings.filterwarnings("ignore", category=IntegrationWarning)

# ===================================================== CONSTANTS ===================================================== #

G = 4.30091e-6  # kpc * (km/s)^2 / Msun
SPARC_COLS = ['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul']

# ===================================================== MODIFIABLE PARAMS ===================================================== #

GALAXY_NAME = 'UGC2885'
PROFILE_TYPE = 'DC14'  # Options: 'NFW', 'Burkert', 'Einasto', 'DC14', 'MOND'
OPT_ALG = 'BFGS'  # Options: 'BFGS', 'TRF', 'MCMC', 'DE', 'PSO', 'BH'
LOSS_FUNC = 'chi_squared'  # Options: 'chi_squared', 'huber', 'cauchy', 'ODR', 'bayesian'

# ===================================================== DATA ===================================================== #

# ==================================================================================
# DATA SOURCE: SPARC (Lelli et al. 2016, AJ, 152, 157)                             #
#   https://astroweb.cwru.edu/SPARC/                                               #
# ==================================================================================

ugc2885 = pd.read_csv('./data/Rotmod_LTG/UGC02885_rotmod.dat', sep=r"\s+", comment='#', header=None, names=SPARC_COLS)

print(f"Data for {GALAXY_NAME}:")

match GALAXY_NAME:
    case 'UGC2885':
        print(ugc2885.head())
        r = ugc2885['Rad'].values  # kpc
        v_obs = ugc2885['Vobs'].values  # km/s
        v_gas = ugc2885['Vgas'].values  # km/s
        v_disk = ugc2885['Vdisk'].values  # km/s
        v_bul = ugc2885['Vbul'].values  # km/s
        sigma = ugc2885['errV'].values  # km/s

has_bulge = np.any(v_bul > 0)

# ===================================================== MODEL ===================================================== #

# Mathematical expansion for dark matter velocity based on NFW profile (cusp)
def NFW_expansion(r, rho_s, r_s):
    return np.sqrt(4 * np.pi * G * rho_s * r_s**3 * (np.log(1 + r/r_s) - (r/r_s)/(1 + r/r_s)) / r)
# Expansion based on Burkert profile (flattened core)
def Burkert_expansion(r, rho_s, r_s):
    M = np.pi * rho_s * r_s**3 * (np.log(1 + (r/r_s)**2) + 2 * np.log(1 + r/r_s) - 2 * np.arctan(r/r_s))
    return np.sqrt(G * M / r)
# Expansion based on Einasto profile ("Shape-Shifting" halo)
def Einasto_expansion(r, rho_s, r_s, alpha):
    h = 2.0/alpha
    a = 3.0/alpha
    x = h*(r/r_s)**alpha
    M = 4 * np.pi * rho_s * (r_s**3) * (np.exp(h) / alpha) * (h**(-a)) * gamma(a) * gammainc(a, x)
    return np.sqrt(G * M / r)

def DC14_expansion(r, rho_s, r_s, X):
    alpha = 2.94 - np.log10((10**(X + 2.33))**-1.08 + (10**(X + 2.33))**2.29)
    beta  = 4.23 + 1.34*X + 0.26*(X**2)
    gamma = -0.06 + np.log10((10**(X + 2.56))**-0.68 + 10**(X + 2.56))

    def integrand(x):
        return (x**2) / ( (x**gamma) * (1 + x**alpha)**((beta - gamma)/alpha) )
    
    v_dm = np.zeros_like(r)
    for i, rad in enumerate(r):
        x = rad / r_s
        # We increase the limit to 200 to give the algorithm room for more subdivisions (where density is extremely high/steep)
        # and we set epsrel=1e-4 to tell it to stop panicking once it reaches 99.99% accuracy
        integral_res = quad(integrand, 0, x, limit=200, epsrel=1e-4)[0]
        M = 4*np.pi*rho_s*(r_s**3)*integral_res
        v_dm[i] = np.sqrt(G * M / rad)
    return v_dm

# Implementation of the rotation curve model, which combines contributions from gas, disk, bulge, and dark matter (using expansion above)
def rot_curve_model(vars, r, vgas, vdisk, vbul):
    if has_bulge:
        Ydisk, Ybul, rho_s, r_s, *extra = vars
    else:
        Ydisk, rho_s, r_s, *extra = vars
        Ybul = 0.0  # No bulge contribution for this galaxy

    rho_s_kpc = rho_s * 1e9 # convert from Msun/pc^3 to Msun/kpc^3

    match PROFILE_TYPE:
        case 'NFW':
            v_dm = NFW_expansion(r, rho_s_kpc, r_s)
        case 'Burkert': 
            v_dm = Burkert_expansion(r, rho_s_kpc, r_s)
        case 'Einasto':
            alpha = extra[0]
            v_dm = Einasto_expansion(r, rho_s_kpc, r_s, alpha)
        case 'DC14':
            X = extra[0]
            v_dm = DC14_expansion(r, rho_s_kpc, r_s, X)
        case 'MOND':
            pass  # TODO: Implement MOND rotation curve calculation

    v_pred = np.sqrt(vgas**2 + Ydisk*vdisk**2 + Ybul*vbul**2 + v_dm**2)
    return v_pred

# Chi-squared function to evaluate the goodness of fit between the observed rotation velocity, 
# and the model's predicted velocity, weighted by the observational uncertainties (sigma)
def chi_squared(vars, r, vgas, vdisk, vbul, v_obs, sigma):
    v_model = rot_curve_model(vars, r, vgas, vdisk, vbul)
    return np.sum(((v_obs - v_model) / sigma)**2)

def loss_function(vars, r, vgas, vdisk, vbul, v_obs, sigma):
    match LOSS_FUNC:
        case 'chi_squared':
            return chi_squared(vars, r, vgas, vdisk, vbul, v_obs, sigma)
        case 'huber':
            pass  # TODO: Implement Huber Loss
        case 'cauchy':
            pass  # TODO: Implement Cauchy Loss
        case 'ODR':
            pass  # TODO: Implement Orthogonal Distance Regression
        case 'bayesian':
            pass  # TODO: Implement Bayesian Inference (e.g., using emcee)

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
# [4] Absolute upper bounds for Y_bulge (~1.2) based on the asymptotic limits      #
#     of Stellar Population Synthesis models for the oldest, most metal-rich       #
#     stellar populations at 3.6um (Schombert, McGaugh, & Lelli,                   #
#     2019, MNRAS, 483, 1496).                                                     #
# [5] Initial guess (0.17) and bounds (0.1, 0.3) for the Einasto index (alpha)     #
#     derived from high-resolution N-body cosmological simulations                 #
#     (Dutton & Macciò, 2014, MNRAS, 441, 3359) and applied to SPARC               #
#     galaxies (Li et al., 2020, ApJS, 247, 31).                                   #
# [6] Initial guess (-2.5) and bounds (-4.5, -1.0) for the stellar-to-halo         #
#     mass ratio (X) governing baryonic feedback in the DC14 profile               #
#     (Di Cintio et al., 2014, MNRAS, 437, 415).                                   #
# ==================================================================================

if has_bulge:
    # Initial guess for the parameters: [Ydisk, Ybul, rho_s, r_s]
    initial_guess = [0.5, 0.7, 0.01, 10.0]
    # Bounds (min, max)
    physical_bounds = [(0.1, 0.8),  # Ydisk
            (0.1, 1.2),  # Ybul
            (1e-4, 1.0),  # rho_s (Msun/kpc^3)
            (0.1, 100.0)]  # r_s (kpc)
else:
    # Initial guess for the parameters: [Ydisk, rho_s, r_s]
    initial_guess = [0.5, 0.01, 10.0]
    physical_bounds = [(0.1, 0.8),  # Ydisk
            (1e-4, 1.0),  # rho_s (Msun/kpc^3)
            (0.1, 100.0)]  # r_s (kpc)

match PROFILE_TYPE:
    case 'Einasto':
        initial_guess.append(0.17)  # initial guess for Einasto alpha parameter
        physical_bounds.append((0.1, 0.3))  # bounds for alpha
    case 'DC14':
        # X=log10​(M∗​/Mhalo​)
        initial_guess.append(-2.5)
        physical_bounds.append((-4.5, -1.0))  # Bounds for X

success = False
results = None

match OPT_ALG:
    # =================================== BFGS =================================== #
    case 'BFGS':
        res = minimize(
            fun=loss_function,
            x0=initial_guess,
            args=(r, v_gas, v_disk, v_bul, v_obs, sigma),
            method = 'L-BFGS-B',
            bounds=physical_bounds
        )
        success = True
        if res.success:
            results = res
        else:
            results = res.message
    # =================================== TRF =================================== #
    case 'TRF':
        pass # TODO: Implement Trust Region Reflective optimization
    # =================================== MCMC =================================== #
    case 'MCMC':
        pass # TODO: Implement Markov Chain Monte Carlo optimization (e.g., using emcee)
    # =================================== DE =================================== #
    case 'DE':
        pass # TODO: Implement Differential Evolution optimization
    # =================================== PSO =================================== #
    case 'PSO':
        pass # TODO: Implement Particle Swarm Optimization
    # =================================== BH =================================== #
    case 'BH':
        pass # TODO: Implement Basin-Hopping optimization

if success:
    print("\n")
    print(results)
    if has_bulge:
        Ydisk_opt, Ybulge_opt, rho_s_opt, r_s_opt, *extra = res.x
        print(f"Optimal Ydisk:  {Ydisk_opt:.3f}")
        print(f"Optimal Ybulge: {Ybulge_opt:.3f}")
    else:
        Ydisk_opt, rho_s_opt, r_s_opt, *extra = res.x
        print(f"Optimal Ydisk:  {Ydisk_opt:.3f}")
        
    print(f"Optimal rho_0:  {rho_s_opt:.4f}")
    print(f"Optimal r_s:    {r_s_opt:.2f}")
    
    match PROFILE_TYPE:
        case 'Einasto':
            print(f"Optimal alpha:  {extra[0]:.3f}")
        case 'DC14':
            print(f"Optimal X:      {extra[0]:.3f}")

    print(f"Final Cost ({LOSS_FUNC}): {res.fun:.2f}")
else:
    print("\nOptimization Failed!")
    print(results)



# TODO: Finish implementation of standard chi-squared minimization using bounded BFGS
# TODO: Implement different opt algs (TRF, MCMC, Dff Evolution, PSO, Basin-Hopping)
# TODO: Implement different loss functions (Huber/Cauchy Loss, ODR, Bayesian Inference)
# TODO: Implement different halo profiles (Burkert, Einasto, DC14, MOND)

# ====================================================== VISUALIZATION ===================================================== #

## Extract the optimized parameters and compute the corresponding rotation curve components for plotting... ##

if has_bulge:
    Ydisk_opt, Ybulge_opt, rho_s_pc_opt, r_s_opt, *extra = res.x
else:
    Ydisk_opt, rho_s_pc_opt, r_s_opt, *extra = res.x
    Ybulge_opt = 0.0

rho_s_opt = rho_s_pc_opt * 1e9 # convert to Msun/kpc^3 for plotting

v_disk_scaled = np.sqrt(Ydisk_opt) * v_disk
v_bul_scaled = np.sqrt(Ybulge_opt) * v_bul
match PROFILE_TYPE:
    case 'NFW':
        v_dm_opt = NFW_expansion(r, rho_s_opt, r_s_opt)
    case 'Burkert':
        v_dm_opt = Burkert_expansion(r, rho_s_opt, r_s_opt)
    case 'Einasto':
        alpha_opt = extra[0]
        v_dm_opt = Einasto_expansion(r, rho_s_opt, r_s_opt, alpha_opt)
    case 'DC14':
        v_dm_opt = DC14_expansion(r, rho_s_opt, r_s_opt, extra[0])
    case 'MOND':
        pass  # TODO: Implement MOND rotation curve calculation

v_tot_opt = np.sqrt(v_gas**2 + v_disk_scaled**2 + v_bul_scaled**2 + v_dm_opt**2)

## Create the plot and display it... ##

# Split plot into two subpolots: data visualtization on the left, and optimized parameters + final cost as text on the right
fig, (ax_plot, ax_text) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})

# Plot data
ax_plot.errorbar(r, v_obs, yerr=sigma, fmt='o', label=r'$V_{total}$ Observed', color='black')
ax_plot.plot(r, v_tot_opt, label=r'$V_{total}$ Model' f'({PROFILE_TYPE})', linewidth=2, color='red')
ax_plot.plot(r, v_gas, 'b-.', label='Gas')
ax_plot.plot(r, v_disk_scaled, 'y--', label=r'Stellar Disk ($\Upsilon_{disk}$ scaled)')

if has_bulge:
    ax_plot.plot(r, v_bul_scaled, 'r:', label=r'Stellar Bulge ($\Upsilon_{bulge}$ scaled)')   

ax_plot.plot(r, v_dm_opt, 'g--', label='Dark Matter Halo')

# Plot formatting
ax_plot.set_title(f'Rotation Curve Decomposition: {GALAXY_NAME}', fontsize=14)
ax_plot.set_xlabel('Radius (kpc)', fontsize=12)
ax_plot.set_ylabel('Velocity (km/s)', fontsize=12)
ax_plot.set_xlim(0, max(r) * 1.05)
ax_plot.set_ylim(0, max(v_obs) * 1.2)
ax_plot.legend(loc='lower right', frameon=False)
ax_plot.grid(True, linestyle=':', alpha=0.6)

# Text Box Stuff
ax_text.axis('off')

param_text = f"Optimized Parameters\n\n"
param_text += f"$\\Upsilon_{{disk}}$ :  {Ydisk_opt:.3f}\n"

if has_bulge:
    param_text += f"$\\Upsilon_{{bulge}}$ :  {Ybulge_opt:.3f}\n"

param_text += f"$\\rho_0$ :  {rho_s_pc_opt:.4f} $M_\\odot/pc^3$\n"
param_text += f"$r_s$ :  {r_s_opt:.2f} kpc\n"

match PROFILE_TYPE:
    case 'Einasto':
        param_text += f"$\\alpha$ :  {alpha_opt:.3f}\n"
    case 'DC14':
        param_text += f"$X$ :  {extra[0]:.3f}\n"

param_text += f"\nFinal Cost ($\\chi^2$) :  {res.fun:.2f}"

ax_text.text(0.1, 0.5, param_text, fontsize=12, va='center', ha='left',
             bbox=dict(boxstyle="round,pad=1", facecolor="#f8f9fa", edgecolor="gray", alpha=0.8))

# display
plt.tight_layout()
plt.show()