#import dependncies and create universe
import MDAnalysis as mda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import curve_fit
import scipy.integrate as integrate
import scipy.special as special
from numpy import log as ln
import math
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis as HBA
from MDAnalysis import Universe

TAU_MAX = 25
CUTOFF = 3.0

helix_names = ['H3_\u03B11', 'H3_\u03B12', 'H3_\u03B13', 
                'H4_\u03B11', 'H4_\u03B12', 'H4_\u03B13',
                'H2A_\u03B11', 'H2A_\u03B12', 'H2A_\u03B13',
	            'H2B_\u03B11', 'H2B_\u03B12', 'H2B_\u03B13', 'H2B_\u03B14']

helix_pos = ['resid 358-372', 'resid 380-407', 'resid 415-425', 
            'resid 460-469', 'resid 479-504', 'resid 512-521', 
            'resid 558-566', 'resid 577-603', 'resid 611-627',
	        'resid 694-704', 'resid 712-739', 'resid 747-757', 'resid 760-779']
# for testing, best to use more frames on less helixes


def create_universe(top_path, traj_path):
    return Universe(top_path, traj_path)

def double_expo(arbitrary_arg, A, tauniverse, B, tau2):
    return A * np.exp(-tau_times / tauniverse) + B * np.exp(-tau_times / tau2)

def get_interactions(first_res, second_res, g_hydrogen, g_acceptors, tau_max_): # calculate hydrogen bonds between selected residues
    '''
        returns [ frame #, donor, hydrogen, acceptor, distance, angle ] for each frame
        example inputs:
            get_interactions('resid 511 to 522', 'resnmae WAT', 'resid 511-522 711-740', 'resnmae WAT')
            get_interactions('resid 511 to 522', 'resid 711 to 740', 'resid 511-522 711-740', 'resnmae WAT')
    '''

    hbonds = HBA(
        universe = universe, 
        update_selections = True,
        d_a_cutoff = CUTOFF,
        d_h_a_angle_cutoff = 150,
        between = [first_res, second_res]
    )

    hbonds.hydrogens_sel = f"({hbonds.guess_hydrogens(g_hydrogen)})"
    hbonds.acceptors_sel = f"({hbonds.guess_acceptors(g_acceptors)})"

    hbonds.run()
    tau, tim = hbonds.lifetime(tau_max = tau_max_)

    return hbonds.results, tau, tim


# create universe
print("Creating universe...")
universe = create_universe('/home/augustine/Nucleosome_system/1kx5/1kx5_0.15M/lifetime/1kx5_015M_OPC.prmtop', '/home/augustine/Nucleosome_system/1kx5/1kx5_0.15M/lifetime/lifetime_1kx5_015M.xtc')

# create files for each helix to save hbond and lifetime results
print("Calculating interactions and lifetimes...")
h = 0
for pos in helix_pos:
    bonds, x, y = get_interactions(pos, 'resname WAT', pos, 'resname WAT', TAU_MAX)

    np.savetxt(f't_{helix_names[h]}.csv', (x, y), delimiter = ',')
    np.savetxt(f'i_{helix_names[h]}.csv', bonds['hbonds'], delimiter = ',')

    h += 1

# find optimal parameters and plot for each helix
print("Finding probabilities...")
                            # - 1 ?
tau_times = np.linspace(0, TAU_MAX, TAU_MAX + 1) * universe.trajectory.dt

for helix in helix_names:
    helix_tau = pd.read_csv(f"t_{helix}.csv", header = None)

    optimizes_params, params_covariance = curve_fit(double_expo, tau_times, helix_tau.iloc[1], p0=[1, 0.5, 1, 2])
    A_, tau1_, B_, tau2_ = optimizes_params

    real_time = np.linspace(min(tau_times), max(tau_times), TAU_MAX + 1)
    probability = double_expo(real_time, A_, tau1_, B_, tau2_)

    plt.plot(real_time, probability)

# plot details
plt.xlabel(r"$\tau\ \rm (ps)$")
plt.ylabel(r"$C(\tau)$")
plt.legend(helix_names, bbox_to_anchor = (1, 1))


# save plot
plt.tight_layout() # so legend is not cut off
plt.savefig("model.png")


print("Done. A figure of all helixes can be found at model.png")
