# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:41:08 2024

#########################################################

**Owner:** Joao Vitor Silva Pontalti - Undergraduate student  
**Advisor:** Eric Brandão Carneiro - Prof. Dr. Eng.
---
## 	**ACOUSTICAL ENGINEERING**  
**UNIVERSIDADE FEDERAL DE SANTA MARIA**

**Last update:** 02/04/2025
---
# Post Processing used for my TCC
@author: joaop

"""

import matplotlib.pyplot as plt
import numpy as np
from controlsair import AirProperties, AlgControls#, add_noise, add_noise2
from sources import Source
from receivers import Receiver
from qterm_estimation import ImpedanceDeductionQterm
from ppro_meas_insitu import InsituMeasurementPostPro
import pytta
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

#%% Name of things
name = 'teste_MicBruel_Focusrite_17042025_1' #'PET_grooved_plate' # 'melamine' #
main_folder = 'C:/Users/joaop/anaconda3/envs/Dataset_JoaoP/TCC_DATASET_JoaoP/DATA2025/'
#%%

# name = 'testing_meas'
# main_folder = 'D:/Work/dev/scanner_meas/meas_scripts'# use forward slash

#%% Intantiate post processin object - it will load the meas_obj
ppro_obj = InsituMeasurementPostPro(main_folder = main_folder, name = name,t_bypass=0)
# source = Source([0, 0, 0.3])
# ppro_obj.meas_obj.source = source

#%% Compute all IR
ppro_obj.compute_all_ir_load(regularization = True,  deconv_with_rec = True, 
                    only_linear_part = True, invert_phase = True)

#%% Load all IR 
ppro_obj.load_irs()

# ht = ppro_obj.meas_obj.IR.plot_time(xLim = (0,1));

#%%

# ppro_obj.receivers.coord

#%% 

# ppro_obj.load_ir_byindex(idir = 1)

# idir = 1

# filename = 'rec' + str(int(idir)) + '_m0.hdf5'
# complete_path = main_folder / name / 'measured_signals'
# med_dict = pytta.load(str(complete_path / filename))
# keyslist = list(med_dict.keys())
# ht = med_dict[keyslist[0]]

#%%
# idir = 1
# ppro_obj.move_ir(idir = idir, c0 = 340, source_coord = ppro_obj.meas_obj.source.coord, 
#                   receiver_coord = ppro_obj.meas_obj.receivers.coord[idir,:],
#                   plot_ir = True, xlims = (-0.1e-3, 2e-3))

# ppro_obj.move_all_ir(c0 = 340)

#%% Lets choose 2 mic arrays

# Select a microphone position and find its relative mic, 
# placed above or below the one. 


# Coords retrieves the coordinates for mic positions. Its a N x 3 matrix
# index is the number of the measurement you want to find

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

#%% Plotting the new scene

#### This here is cmpletely taken from the sequential_measurement code.
#### I've just adapted for us to have a better visualization of the choosen mic positions

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

# Plot source (Se houver)
if ppro_obj.meas_obj.source is not None:
    ax.scatter(ppro_obj.meas_obj.source.coord[0, 0],
               ppro_obj.meas_obj.source.coord[0, 1],
               ppro_obj.meas_obj.source.coord[0, 2],
               s=200, marker='*', color='red', alpha=0.5)

# Plot receivers (microfones)
for r_coord in range(ppro_obj.meas_obj.receivers.coord.shape[0]):
    ax.scatter([ppro_obj.meas_obj.receivers.coord[r_coord, 0]],
               [ppro_obj.meas_obj.receivers.coord[r_coord, 1]],
               [ppro_obj.meas_obj.receivers.coord[r_coord, 2]],
               marker='o', s=12, color='blue', alpha=0.7)

# Plot selected pair position
for i, (par_index, par) in enumerate(pares, start=1):
    ax.scatter(par[0], par[1], par[2], s=50, color='red', label=f'Par {i}' if i == 1 else "")

ax.set_title("Measurement scene - 2 mics selected")
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

### NOTE: From here on, Im picking the index and the par_index as the 
### idir values for plotting. This adjust all the subsequent plotting to be made
### for the choosen pair of microphones.


tlims = (0e-3, 20e-3)
fig, ax = plt.subplots(1, figsize = (8,6), sharex = False)
ppro_obj.plot_ir(ax, idir = index, normalize = True, xlims = tlims, windowed = False)
ppro_obj.plot_ir(ax, idir = par_index, normalize = True, xlims = tlims, windowed = False)
ax.set_xlabel("Time [s]", fontsize = 18)
ax.set_ylabel("Amplitude [-]", fontsize = 18)
ax.grid()

#%% Set the Adrienne window and apply it on IRs - plot the result
# adrienne = ppro_obj.set_adrienne_win(tstart = 20e-3, dt_fadein = 1e-3, t_cutoff = 27e-3, dt_fadeout = 2e-3)

# t_cutoff =  11e-3
adrienne = ppro_obj.set_adrienne_win(tstart = 0e-3, dt_fadein = .4e-3, t_cutoff = 13e-3, dt_fadeout = 1.3e-3)
ppro_obj.apply_window()

rec_index = 0
ppro_obj.ir_raw_vs_windowed(idir = index, xlims = tlims)

ppro_obj.frf_raw_vs_windowed(idir = par_index, ylims = (-100, 0))


#%% Reset frequency resolution
ppro_obj.reset_freq_resolution(freq_init = 100, freq_end = 4000, delta_freq = 10)

#%% Estimate absorption and plot
air = AirProperties(c0 = 343.0, rho0 = 1.21,)
controls = AlgControls(c0 = air.c0, freq_vec = ppro_obj.freq_Hw) 

h_pp = ImpedanceDeductionQterm(p_mtx=ppro_obj.Hww_mtx, controls=controls, 
                               receivers=ppro_obj.meas_obj.receivers, 
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
