# -*- coding: utf-8 -*-
"""
#########################################################

**Owner:** Joao Vitor Silva Pontalti - Undergraduate student  
**Advisor:** Eric Brandão Carneiro - Prof. Dr. Eng.
---
 	  **ACOUSTICAL ENGINEERING (EAC)**  
**UNIVERSIDADE FEDERAL DE SANTA MARIA (UFSM)**

**Last update:** 22/02/2025
---
@author: joaop

Created this code totally based on the minimum_ppro code, but added  a few lines
for checking individual pairs of microphones to be analyzed as 2 mic array.

Added extra parts regargind reconstruction of the sound field above the sample
"""
#%% Import the important things

import matplotlib.pyplot as plt
import numpy as np
import pytta

from controlsair import AirProperties, AlgControls#, add_noise, add_noise2
from sources import Source
from receivers import Receiver

from ppro_meas_insitu import InsituMeasurementPostPro
from joaop_functions import sph2cart, format_meas_name, select_mics

from decomposition_ev_ig import DecompositionEv2, ZsArrayEvIg, filter_evan
from qterm_estimation import ImpedanceDeductionQterm
from scipy.interpolate import griddata

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
    
from decomp_quad_v2 import Decomposition_QDT
from decomp2mono import Decomposition_2M  # Monopoles

#%% Name of things
# NOTE: Dont forget the forward slash

name = 'ThickFoam_L60cm_d135mm_s100cm_2planar_18022025' #'PET_grooved_plate' # 'melamine' #
main_folder = 'C:/Users/joaop/anaconda3/envs/Dataset_JoaoP/TCC_DATASET_JoaoP/DATA2025/'

#%% Instantiate post processin object - it will load the meas_obj

##########CRIA O OBJ
ppro_obj = InsituMeasurementPostPro(main_folder = main_folder, name = name,t_bypass=0)
# source = Source([0, 0, 0.3])
# ppro_obj.meas_obj.source = source

#%% Format title, extract infos for plotting 
index = 25
#joaop_functions
formatted_name = format_meas_name(name, basic_infos=False) # joaop_functions
#%% #joaop_functions
par_index = select_mics(ppro_obj = ppro_obj, index = index, baffle_size=1.5, plot=True) 

#%% Compute all IR
ppro_obj.compute_all_ir_load(regularization = True,  deconv_with_rec = True, 
                   only_linear_part = True)

#%% Load all IR 
ppro_obj.load_irs()

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
adrienne = ppro_obj.set_adrienne_win(tstart = 1e-3, dt_fadein = 1e-3, t_cutoff = 11e-3, dt_fadeout = 4.5e-3)
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

# GUS VAI ATÉ AQUI


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
receivers = ppro_obj.meas_obj.receivers 
z_top = 0.01
theta = 0 # just so the source will be pointed to the ground
# receiv = Receiver.hemispherical_array

zs_ded = DecompositionEv2(p_mtx = ppro_obj.Hww_mtx, controls = controls, material = None, delta_x = 0.05, delta_y = 0.05,
                      receivers = ppro_obj.meas_obj.receivers, regu_par = 'gcv')
#%%
zs_ded.prop_dir(n_waves = 642, plot = False)
#%%
zs_ded.pk_tikhonov_ev_ig(f_ref=1, f_inc=1, factor=1.5, zs_inc = z_top, zs_ref = 0.0,
                          num_of_grid_pts = 3*int(zs_ded.prop_waves_dir.n_prop**0.5), plot_l=False, method = 'Tikhonov')
#%%
zs_ded.reconstruct_p(receivers, compute_inc_ref = False)

#%% Directivity plot
freq2plot = 500
dinrange = 20
# set plotting object
x, y, z = sph2cart(2, np.deg2rad(15), np.deg2rad(-105))
eye = dict(x=x, y=y, z=z)

fig2, trace2 = zs_ded.plot_directivity(freq = freq2plot, color_method = 'dB', radius_method = 'dB',
                                    dinrange = dinrange, color_code = 'jet', view = 'iso_z', eye=eye,  
                                    renderer = "browser", true_directivity = False)
fig2.show()


#%% Compute your sound field reconstruction (TAKE SOME TIME TO DO IT)

## CALCULATING TICKONOV INVERSION  WITH EVANESCENT wAVES AND IRREGUlAR GRID

# zs_ded.pk_tikhonov_ev_ig(f_ref = 1.0, f_inc = 1.0, factor = 1.5, zs_ref = 0.0, zs_inc = 1.5, num_of_grid_pts = None,plot_l = False, method = 'Tikhonov')
# zs_ded.pk_tikhonov_ev_ig(f_ref=1, f_inc=1, factor=1.5, z0 = 40/1000+20/1000, zref = 0.0, plot_l=False, method = 'Tikhonov')
# zs_ded.zs(Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21, theta = [np.deg2rad(theta)]);

#%%
zs_ded.plot_inc_ref_pkmap(freq =1000, db = False, dinrange = 15,
    color_code = 'viridis', figsize=(10, 5), fine_tune_subplt = [0.1, 0, 0.9, 0.99])


#%%
zs_ded.plot_ref_pkmap()
#%%

zs_ded.plot_ref_pkmap()

zs_ded.plot_directivity(freq = 1000, dinrange = 20,
        save = False, fig_title = '', path = '', fname='', color_code = 'viridis',
        true_directivity = True, dpi = 600, figsize=(8, 8), fileformat='png',
        color_method = 'dB', radius_method = 'dB',
        view = 'iso_z', eye = None, renderer = 'notebook',
        remove_axis = False)

#%% Alpha estimation

# test = zs_ded.vp_surf()        # using a decomposition_ev_ig function
#                                 # 'reconstruct the surface reflection coef in z = 0.0 and 
#                                 # and estimate the abs.coef.'
                                
# test2 = zs_ded.alpha_from_pk()   # using a decomposition_ev_ig function
#                                 # 'calculate the abs.coef from wavenumber spectra'
                                
#%%
# Instatiate

num_gauss_pts = 25
a = 0 # lower limit of integral (always zero usually - we left it open for tests)
b = 30 # upper limit of integral (truncation)

decomp_qdt_gleg = Decomposition_QDT(p_mtx=ppro_obj.Hww_mtx, controls=controls,
    receivers=ppro_obj.meas_obj.receivers, source_coord=ppro_obj.meas_obj.source.coord[0],
    quad_order=num_gauss_pts, a = a, b = b, 
    retraction = 0, image_source_on = True, regu_par = 'gcv')

# Choose sampling scheme
decomp_qdt_gleg.gauss_legendre_sampling()

# Solve
decomp_qdt_gleg.pk_tikhonov(plot_l=False, method='Tikhonov')
#decomp_qdt_gleg.least_squares_pk()
# Reconstruct surface impedance
decomp_qdt_gleg.zs(Lx=0.1, n_x=21, Ly=0.1, n_y=21, theta=[0], avgZs=True);  # Zs

#%% Plotting the alphas obtained 'til this point

plt.figure(figsize = (8,4))
plt.semilogx(PWA_2mic[:,0], PWA_2mic[:,1], '-g', label = '2 mic', alpha = 0.7)
# plt.semilogx(zs_ded.controls.freq, test, '-ob', label = 'Array -> vp_Surf', alpha = 0.4)
# plt.semilogx(zs_ded.controls.freq, test2, '-or', label = 'Array -> alpha_from_pk', alpha = 0.4)
plt.semilogx(zs_ded.controls.freq,zs_ded.alpha.flatten(), 'k', label = 'NAH', alpha = 0.7)
plt.semilogx(zs_ded.controls.freq,decomp_qdt_gleg.alpha.flatten(), 'r', label = 'DCISM', alpha = 0.7)


plt.legend()
plt.grid()
plt.ylim((-0.2, 1.2))
plt.xlim((125, 4000))
plt.xticks(ticks = [125,250,500,1000,2000,4000], labels = ['125','250','500','1000','2000','4000'], rotation = 0)
plt.xlabel(r'Frequency [Hz]')
plt.ylabel(r'$\alpha$ [-]')
plt.title("Elev. angle: {:.1f}$^\circ$".format(theta))
plt.tight_layout();

#%% Wavenumber Spectrum Plot

freq = [1500]
dinrange = 15
# freq = [500,1000,2000]

for f in freq:
        zs_ded.plot_pkmap_v2(freq = f, db = True, dinrange = 15, color_code='inferno', fileformat = 'pdf', 
    fig_title=f"{f} Hz / dinrange = {dinrange}", plot_incident = False, save = False, path = 'C:/Users/joaop/anaconda3/envs/Dataset_JoaoP/TCC_DATASET_JoaoP/PLOTS_2Planar_reconstruct', fname = '', dpi=600, figsize = (5.5,4))
    
    