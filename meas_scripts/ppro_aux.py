# -*- coding: utf-8 -*-
"""
#########################################################

**Owner:** Joao Vitor Silva Pontalti - Undergraduate student  
**Advisor:** Eric Brandão Carneiro - Prof. Dr. Eng.
---
 	  **ACOUSTICAL ENGINEERING (EAC)**  
**UNIVERSIDADE FEDERAL DE SANTA MARIA (UFSM)**

**Last update:** 28/01/2025
---
@author: joaop

Examples sent by professor to help me with the plotting and stuff
"""
#%% Import the important things

import matplotlib.pyplot as plt
import numpy as np
import pytta

from controlsair import AirProperties, AlgControls#, add_noise, add_noise2
from sources import Source
from receivers import Receiver

from ppro_meas_insitu import InsituMeasurementPostPro

from decomposition_ev_ig import DecompositionEv2, ZsArrayEvIg, filter_evan
from qterm_estimation import ImpedanceDeductionQterm
from scipy.interpolate import griddata

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
    
from decomp_quad_v2 import Decomposition_QDT
from decomp2mono import Decomposition_2M  # Monopoles

#%%
name = 'AbsTriangPurple_L60cm_d5cm_s100cm_2planar_14012025' #'PET_grooved_plate' # 'melamine' #
main_folder = 'C:/Users/joaop/anaconda3/envs/Dataset_JoaoP/TCC_DATASET_JoaoP/DATA2025/'

#%% Instantiate post processin object - it will load the meas_obj

ppro_obj = InsituMeasurementPostPro(main_folder = main_folder, name = name,t_bypass=0)
# source = Source([0, 0, 0.3])
# ppro_obj.meas_obj.source = source

#%% Compute all IR
ppro_obj.compute_all_ir_load(regularization = True,  deconv_with_rec = True, 
                   only_linear_part = True)

#%% Load all IR 
ppro_obj.load_irs()

#%% Lets choose a mic position

"""
# Selecting a Microphone Position and Finding its Relative Pair

This function helps you select a microphone position and find its relative microphone, 
which is positioned either above or below the one you initially chose.

## Parameters:
- **Coords**: 
  A matrix of size N x 3 containing the spatial coordinates of the microphone positions.
- **index**: 
  The index of the measurement you want to examine. It must be an integer within the range 
  of available measurements. For example, in an array with 128 measurement points, you can 
  choose an index from 0 to 127.

## Usage Recommendation:
1. Insert an `index` (e.g., 42) to find its relative pair.
2. Verify the result by using the relative index obtained as the new input. 
   This ensures the function correctly identifies matching microphone pairs.

This step-by-step process helps confirm the consistency of the pairing logic.
"""

coords = ppro_obj.meas_obj.receivers.coord  # objects inside objects
index = 0  # Choose a measurement you want to see
coord_index = coords[index] # tace

pares = [] # creating a vec pares

# Find the pais that vary only the Z axis
for i, coord in enumerate(coords): # for all the enumerated coord lines
    if i != index:  # the index must be different (just a way to avoid redundancy)
        if coord[0] == coord_index[0] and coord[1] == coord_index[1] and coord[2] != coord_index[2]:
            pares.append((i, coord))  # save the index and the coordinate that 
                                      # suits the conditions 

# Print the coordinates of the choosen measurement and its relative 
print(f"Mic reference (index={index}): {coord_index}")
for i, (par_index, par) in enumerate(pares, start=1):
    print(f"Par {i} - Índice: {par_index}, Coordenada: {par}")

##%% Plotting the new scene

'''
#### This here is cmpletely taken from the sequential_measurement code.
#### I've just adapted for us to have a better visualization of the choosen mic positions
#### The only purpose of this section is aesthetics
'''
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

list_of_sample_verts = []
# Sample top
list_of_sample_verts.append(np.array([[-0.6/2, -0.6/2, 0.0],
                                      [0.6/2, -0.6/2, 0.0],
                                      [0.6/2, 0.6/2, 0.0],
                                      [-0.6/2, 0.6/2, 0.0]]))
# lx 1
list_of_sample_verts.append(np.array([[-0.6/2, -0.6/2, 0.0],
                                      [0.6/2, -0.6/2, 0.0],
                                      [0.6/2, -0.6/2, -0.1],
                                      [-0.6/2, -0.6/2, -0.1]]))
# lx 2
list_of_sample_verts.append(np.array([[-0.6/2, 0.6/2, 0.0],
                                      [-0.6/2, 0.6/2, -0.1],
                                      [0.6/2, 0.6/2, -0.1],
                                      [0.6/2, 0.6/2, 0.0]]))
# ly 1
list_of_sample_verts.append(np.array([[-0.6/2, -0.6/2, 0.0],
                                      [-0.6/2, -0.6/2, -0.1],
                                      [-0.6/2, 0.6/2, -0.1],
                                      [-0.6/2, 0.6/2, 0.0]]))
# ly 2
list_of_sample_verts.append(np.array([[0.6/2, -0.6/2, 0.0],
                                      [0.6/2, 0.6/2, 0.0],
                                      [0.6/2, 0.6/2, -0.1],
                                      [0.6/2, -0.6/2, -0.1]]))

# Plotting the sample vertices
for jv in np.arange(len(list_of_sample_verts)):
    verts = [list(zip(list_of_sample_verts[jv][:, 0],
                     list_of_sample_verts[jv][:, 1], list_of_sample_verts[jv][:, 2]))]
    collection = Poly3DCollection(verts,
                                  linewidths=0.5, alpha=0.5, edgecolor='tab:blue', zorder=2)
    collection.set_facecolor('tab:blue')
    ax.add_collection3d(collection)

# Baffle
baffle_size = 1.2
baffle = np.array([[-baffle_size/2, -baffle_size/2, -0.1],
                   [baffle_size/2, -baffle_size/2, -0.1],
                   [baffle_size/2, baffle_size/2, -0.1],
                   [-baffle_size/2, baffle_size/2, -0.1]])

verts = [list(zip(baffle[:, 0], baffle[:, 1], baffle[:, 2]))]
collection = Poly3DCollection(verts,
                              linewidths=0.5, alpha=0.5, edgecolor='grey', zorder=2)
collection.set_facecolor('grey')
ax.add_collection3d(collection)

# Plot source
if ppro_obj.meas_obj.source is not None:
    ax.scatter(ppro_obj.meas_obj.source.coord[0, 0],
               ppro_obj.meas_obj.source.coord[0, 1],
               ppro_obj.meas_obj.source.coord[0, 2],
               s=200, marker='*', color='red', alpha=0.5)

# Plot receivers
for r_coord in range(ppro_obj.meas_obj.receivers.coord.shape[0]):
    ax.scatter([ppro_obj.meas_obj.receivers.coord[r_coord, 0]],
               [ppro_obj.meas_obj.receivers.coord[r_coord, 1]],
               [ppro_obj.meas_obj.receivers.coord[r_coord, 2]],
               marker='o', s=12, color='blue', alpha=0.7)

# ADDED: Plot selected pair position 
for i, (par_index, par) in enumerate(pares, start=1):
    ax.scatter(par[0], par[1], par[2], s=50, color='red', label=f'Par {i}' if i == 1 else "")

ax.set_title(f"Measurement scene - 2 mics selected {index} e {par_index}")
ax.set_xlabel(r'$x$ [m]')
ax.set_ylabel(r'$y$ [m]')
ax.set_zlabel(r'$z$ [m]')
ax.grid(False)
ax.set_xlim((-baffle_size/2, baffle_size/2))
ax.set_ylim((-baffle_size/2, baffle_size/2))
ax.set_zlim((-0.1, 1.0))
ax.view_init(elev=30, azim=45)
plt.tight_layout()


#%% Plot - check for good health

'''
### NOTE: From here on, Im picking the index and the par_index as the 
### idir values for plotting. This adjust all the subsequent plotting to be made
### for the choosen pair of microphones.
'''

tlims = (0e-3, 20e-3)
fig, ax = plt.subplots(1, figsize = (8,6), sharex = False)
ppro_obj.plot_ir(ax, idir = index, normalize = True, xlims = tlims, windowed = False)
ppro_obj.plot_ir(ax, idir = par_index, normalize = True, xlims = tlims, windowed = False)
ax.grid()

#%% Set the Adrienne window and apply it on IRs - plot the result
# adrienne = ppro_obj.set_adrienne_win(tstart = 20e-3, dt_fadein = 1e-3, t_cutoff = 27e-3, dt_fadeout = 2e-3)
adrienne = ppro_obj.set_adrienne_win(tstart = 1e-3, dt_fadein = 1.5e-3, t_cutoff = 12e-3, dt_fadeout = 2e-3)
ppro_obj.apply_window()

# rec_index = 15
ppro_obj.ir_raw_vs_windowed(idir = index, xlims = tlims)
# ppro_obj.frf_raw_vs_windowed(idir = par_index, ylims = (-100, 0))

#%% Reset frequency resolution
ppro_obj.reset_freq_resolution(freq_init = 100, freq_end = 4000, delta_freq = 10)


#%% Estimate absorption and plot
air = AirProperties(c0 = 343.0, rho0 = 1.21,)
controls = AlgControls(c0 = air.c0, freq_vec = ppro_obj.freq_Hw) 
ids = [index, par_index]
recs = Receiver()
recs.coord = ppro_obj.meas_obj.receivers.coord[ids,:]

h_pp = ImpedanceDeductionQterm(p_mtx=ppro_obj.Hww_mtx[ids,:], controls=controls, 
                               receivers=recs, 
                               source=ppro_obj.meas_obj.source)
h_pp.pw_pp()
h_pp.pwa_pp()
#h_pp.zq_pp(h_pp.Zs_pwa_pp, tol = 1e-6, max_iter = 40);

plt.figure()
plt.semilogx(h_pp.controls.freq, h_pp.alpha_pw_pp, label = 'PW')
plt.semilogx(h_pp.controls.freq, h_pp.alpha_pwa_pp, label = 'PWA')
#plt.semilogx(h_pp.controls.freq, h_pp.alpha_q_pp, label = 'q-term')
plt.legend()
plt.ylim((-0.4, 1.0))
plt.xlim((100, 4000))
plt.grid()
plt.xticks(ticks = [125,250,500,1000,2000,4000], labels = ['125','250','500','1000','2000','4000'], rotation = 0)
plt.xlabel(r'Frequency [Hz]')
plt.ylabel(r'$\alpha$ [-]')
plt.tight_layout();

PWA_2mic = np.column_stack((h_pp.controls.freq, h_pp.alpha_pwa_pp))

#%%

################################################################################
'''                       
    ITS RECONSTRUCTION TIME BABE
'''
################################################################################

#%% Setting basics

''' 
    We utilize here the same data from previous steps, but now going further into
    the DSP behind reconstructing sound fields.
'''

#%%

num_prop_waves = 642
z_top = 0.01
line_array_2 = ppro_obj.meas_obj.receivers 


## Expansão em ondas planas
ded_150 = DecompositionEv2(p_mtx = ppro_obj.Hww_mtx, controls = controls, receivers = ppro_obj.meas_obj.receivers,
                       delta_x = 0.04, delta_y = 0.04, regu_par = 'gcv')
ded_150.prop_dir(n_waves = num_prop_waves, plot = False)
ded_150.pk_tikhonov_ev_ig(f_ref=1, f_inc=1, factor=1.5, zs_inc = z_top, zs_ref = 0.0,
                          num_of_grid_pts = 3*int(ded_150.prop_waves_dir.n_prop**0.5), plot_l=False, method = 'Tikhonov')
ded_150.reconstruct_p(line_array_2, compute_inc_ref = False)
#%%

def sph2cart(r, theta, phi):
    """
    Convert spherical coordinates to Cartesian coordinates.

    Parameters:
    r (float): Radius
    theta (float): Inclination (elevation) in radians
    phi (float): Azimuth in radians

    Returns:
    tuple: (x, y, z) Cartesian coordinates
    """
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)
    return x, y, z


## #%% Directivity plot
freq2plot = 2000
dinrange = 45
# set plotting object
x, y, z = sph2cart(2, np.deg2rad(15), np.deg2rad(-105))
eye = dict(x=x, y=y, z=z)

fig2, trace2 = ded_150.plot_directivity(freq = freq2plot, color_method = 'dB', radius_method = 'dB',
                                    dinrange = dinrange, color_code = 'jet', view = 'iso_z', eye=eye,  
                                    renderer = "browser", true_directivity = False)
fig2.show()

#%%
'''
I need to understand how to process ded_flat
'''
#%%
## Plot de mapas kx, ky - incidente e refletido "ded_flat" é um objeto de expansão em ondas plnas
dinrange = 15
color_code = 'inferno'
fine_tune_subplt = [0.08, 0.1, 0.9, 0.9]
base_size = 1.5
colorbar_ticks = np.arange(-dinrange, 3, 3)
colorbar_label = r'$|\bar{P}(k_x, k_y)|$ [dB]'
n_rows = 4
side_text = ['Flat (Inc.)', 'Flat (Dif.)', '1-Sphere (Dif.)', '4-Spheres (Dif.)']
fig, axs = plt.subplots(n_rows, len(freq2plot), figsize = (len(freq2plot)*base_size, n_rows*base_size),
                        sharex = True, sharey = True)
for col in range(len(freq2plot)):
    _, _ = ded_flat.plot_inc_pkmap(axs[0, col], freq = freq2plot[col], db = True,
                            dinrange = dinrange, color_code = color_code)
    _, _ = ded_flat.plot_ref_pkmap(axs[1, col], freq = freq2plot[col], db = True,
                            dinrange = dinrange, color_code = color_code)
    _, _ = ded_1sph.plot_ref_pkmap(axs[2, col], freq = freq2plot[col], db = True,
                            dinrange = dinrange, color_code = color_code)
    _, cbar = ded_4sph.plot_ref_pkmap(axs[3, col], freq = freq2plot[col], db = True,
                            dinrange = dinrange, color_code = color_code)
    axs[n_rows-1, col].set_xlabel(r'$k_x$ [rad/m]')
    axs[0, col].set_title('{:.1f} kHz'.format(freq2plot[col]/1000))
   
for row in range(n_rows):
    axs[row, 0].set_ylabel(r'$k_y$ [rad/m]')
    # Add title to the right side
    axs[row, len(freq2plot)-1].text(1.1, 0.5, side_text[row], va='center', ha='center',
            rotation=90, transform=axs[row, len(freq2plot)-1].transAxes)

cbar_ax_start = 0.2
cbar_ax = fig.add_axes([0.95, cbar_ax_start, 0.01, 1-2*cbar_ax_start])
fig.colorbar(cbar, cax = cbar_ax, shrink = 0.7, ticks = colorbar_ticks,
             label = colorbar_label)
fig.subplots_adjust(left = fine_tune_subplt[0], bottom = fine_tune_subplt[1],
                    right = fine_tune_subplt[2], top = fine_tune_subplt[3], hspace = 0.03, wspace = 0.03)
