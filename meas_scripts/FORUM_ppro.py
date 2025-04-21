# -*- coding: utf-8 -*-
"""
#########################################################

**Owner:** Joao Vitor Silva Pontalti - Undergraduate student  
**Advisor:** Eric Brandão Carneiro - Prof. Dr. Eng.
---
 	  **ACOUSTICAL ENGINEERING (EAC)**  
**UNIVERSIDADE FEDERAL DE SANTA MARIA (UFSM)**

**Last update:** 18/04/2025
---
@author: joaop

Post processing routine created for better processing DATA for the upcoming 
Forum Acusticum. The idea here is trying to optimize my ppro procedures.
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
import numpy as np
import pytta
import os

from controlsair import AirProperties, AlgControls#, add_noise, add_noise2
from sources import Source
from receivers import Receiver

from ppro_meas_insitu import InsituMeasurementPostPro
from joaop_functions import sph2cart, format_meas_name, select_mics, list_folders,meas_info

from decomposition_ev_ig import DecompositionEv2, ZsArrayEvIg, filter_evan
from qterm_estimation import ImpedanceDeductionQterm
from scipy.interpolate import griddata

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

from decomp_quad_v2 import Decomposition_QDT
from decomp2mono import Decomposition_2M  # Monopoles

#%%  set main folder and array type
main_folder = "C:/Users/joaop/anaconda3/envs/Dataset_JoaoP/TCC_DATASET_JoaoP/DATA2025/" 
keyword = "PETWool_Lx120cm_Ly60cm_Lz4cm_doubleplanar_50x50x2cm_288pts_19042025_BKMic"

#%%  list and select folder meas
meas_names = list_folders(main_folder, keyword)

#%% Create a list of ppro_obj for post processing

##########  It is important to know the order inside this list
         # if you want to assess data individualy 

lista_ppro = [] ### create lista_ppro, list containing all ppro_obj
                ### acess each ppro_obj by index (i.e. lista_ppro[1])

for name in meas_names:
    ppro_obj = InsituMeasurementPostPro(main_folder = main_folder, name = name, t_bypass = 0)
    lista_ppro.append(ppro_obj)
    
######### Comment below if already computed the IR
    ppro_obj.compute_all_ir_load(regularization = True,  deconv_with_rec = True, 
                    only_linear_part = True, invert_phase=True) 
    
    ppro_obj.load_irs()

#%% Format title, extract infos for plotting 

# info_list =[]
# for i, ppro_obj in enumerate(lista_ppro):
#     info = meas_info(meas_name = meas_names[0],basic_infos=False) # needs a re-do
#     info_list.append(info)

#%% Select Mics
index = 122

#%
title_name = []

# par_index = 

for i, ppro_obj in enumerate(lista_ppro):
    if not any(kw in keyword for kw in ["2planar", "2mics"]): # kw é a variável
        par_index = index + 1
        title = select_mics(ppro_obj = ppro_obj, index = index, par_index = par_index, baffle_size=1.5, plot=True)
        print(f"\n {title} Sensor array shape: {lista_ppro[i].meas_obj.receivers.coord.shape}")
    else:        
        par_index, title = select_mics(ppro_obj = ppro_obj, index = index, par_index = None, baffle_size=1.5, plot=True)   
        title_name.append(title)
        print(f"\n {title} Sensor array shape: {lista_ppro[i].meas_obj.receivers.coord.shape}")

#%% Check loaded IRs
#joaop_functions
# formatted_name = format_meas_name(meas_names[1], basic_infos=True) # joaop_functions

#%% Check a IR from each measurement

for i, ppro_obj in enumerate(lista_ppro):
    tlims = (0e-3, 20e-3)
    fig, ax = plt.subplots(1, figsize = (8,6), sharex = False)
    lista_ppro[i].plot_ir(ax, idir = index, normalize = True, xlims = tlims, windowed = False)
    lista_ppro[i].plot_ir(ax, idir = par_index, normalize = True, xlims = tlims, windowed = False)
    ax.set_title(f'{meas_names[i]}\n mic {index} e {par_index}')
    ax.grid()

#%% Adrienne Window

# plt.fig()
for i, ppro_obj in enumerate(lista_ppro):
    adrienne = lista_ppro[i].set_adrienne_win(tstart = 0.5e-3, dt_fadein = 0.5e-3, t_cutoff = 12.6e-3, dt_fadeout = 1e-3)
    lista_ppro[i].apply_window()

# rec_index = 15
    lista_ppro[i].ir_raw_vs_windowed(idir = index, xlims = tlims)
    plt.title(f'{meas_names[i]}\n mic {index} e {par_index}')
    # ppro_obj.frf_raw_vs_windowed(idir = par_index, ylims = (-100, 0))
    # plt.title(f'{meas_names[i]}')

#%% Reset frequency resolution

####### PAY ATTENTION TO THE -> delta_freq <- PARAMETER
      #     It will determine the time for processing in the next steps.     

delta_freq = 30

for i, ppro_obj in enumerate(lista_ppro):
    lista_ppro[i].reset_freq_resolution(freq_init = 100, freq_end = 4000, delta_freq = delta_freq)

#%% Absorption Coef 2mics

pwa_results = [] # Save the vectors for each ppro_obj in lista_ppro

for i, ppro_obj in enumerate(lista_ppro): 
    air = AirProperties(c0 = 343.0, rho0 = 1.21,) # air velocity, air density
    controls = AlgControls(c0 = air.c0, freq_vec = lista_ppro[i].freq_Hw) # 
    ids = [index, par_index]
    recs = Receiver()
    recs.coord = lista_ppro[i].meas_obj.receivers.coord[ids,:]
    
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
    plt.ylim((-0.2, 1.2))
    plt.xlim((100, 4000))
    plt.grid()
    plt.xticks(ticks = [125,250,500,1000,2000,4000], labels = ['125','250','500','1000','2000','4000'], rotation = 0)
    plt.title(f'{meas_names[i]}')
    plt.xlabel(r'Frequency [Hz]')
    plt.ylabel(r'$\alpha$ [-]')
    plt.tight_layout();

    pwa_2mic = np.column_stack((h_pp.controls.freq, h_pp.alpha_pwa_pp))
    pwa_results.append(pwa_2mic)  # Armazenar para cada ppro_obj
    
#%% RECONSTRUCTION TIME

###############################

#           RECONSTRUCTION TIME

###############################

#%% Setting up plane wave expansion
'''
        In here, we setup the inverse problem of a plane wave expansion 
    considering propagating and evanescent waves in the sound field.
        Check the basic theory for a better understanding of each parameter.
'''

# receivers = lista_ppro[i].meas_obj.receivers 
# receiv = Receiver.hemispherical_array

z_top = 0.045 # The location of the source plane for the incident sound field
z_bottom = 0.01
theta = 0 # just so the source will be pointed to the ground
list_zs_ded1 = []
alpha_NAH = []

for i, ppro_obj in enumerate(lista_ppro):
    receivers = lista_ppro[i].meas_obj.receivers # atributes to a variable
    
    zs_ded = DecompositionEv2(p_mtx = lista_ppro[i].Hww_mtx, controls = controls, material = None,
                              delta_x = 0.05, delta_y = 0.05,
                              receivers = lista_ppro[i].meas_obj.receivers,
                              regu_par = 'gcv')
    
    #%
    list_zs_ded1.append(zs_ded)  # Adiciona o objeto à lista
    #%
    zs_ded.prop_dir(n_waves = 642, plot = False) # Constrói array esférico de ondas planas propagantes
    #%
    #% Calculate Tikhonov Reconstruct Pressure
    zs_ded.pk_tikhonov_ev_ig(f_ref=1, f_inc=1, factor=1.5, 
                             zs_inc = z_top, zs_ref = z_bottom,
                             num_of_grid_pts = 3*int(zs_ded.prop_waves_dir.n_prop**0.5), 
                             plot_l=False, method = 'Tikhonov')
    
    #% Reconstruct pressure 
    zs_ded.reconstruct_p(receivers, compute_inc_ref = False)
    
    # -> CHECK-POINT SAVE <-
    # zs_ded.save(filename = 'recon_' + f'{meas_names[i]}_deltafreq{delta_freq}', path = 'C:/Users/joaop/anaconda3/envs/Dataset_JoaoP/TCC_DATASET_JoaoP/reconstructed_data/')
    
    alpha = zs_ded.alpha_from_pk() # estimate alpha from the wavenumber spk
    
    alpha_NAH.append(alpha) # save the date (alphas)
    
#%% Alpha estimation for NAH
# zs_array = ZsArrayEvIg(p_mtx=lista_ppro[0].Hww_mtx, controls = controls,
#                        material=None, receivers=lista_ppro[0].meas_obj.receivers, regu_par='GCV')

#%%
# a.alpha_pk
#%%
# alpha = a.zs(Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21, theta = [0], avgZs = True)

#%% Load reconstructed sound fields


#%% Plot directivity
'''
        The sph2cart function is used here. We plot the directivity for
    a given frequency. Larger dinrange == large side lobes.
'''
freq2plot =1500
dinrange = 45

#%
# set plotting object
x, y, z = sph2cart(2, np.deg2rad(15), np.deg2rad(-105))
eye = dict(x=x, y=y, z=z)

fig2, trace2 = list_zs_ded1[0].plot_directivity(freq = freq2plot, color_method = 'dB', radius_method = 'dB',
                                    dinrange = dinrange, color_code = 'plasma', view = 'iso_z', eye=eye,  
                                    renderer = "browser", true_directivity = False)
fig2.show()

#%% Plot reflected PK-map one subplot

freq = [500,1500,2500]
dinrange = 20

for i, zs_ded in enumerate(list_zs_ded1): # over all elements of reconstruct list
    fig, axes = plt.subplots(1, len(freq), figsize=(4.5 * len(freq), 4.5))
    axes = np.atleast_1d(axes)  # garante que axes seja sempre iterável
    
    for ax, f in zip(axes, freq):          # over all frequencies
        zs_ded.plot_ref_pkmap(ax = ax, freq = f, db = True, dinrange = dinrange,
            color_code = 'jet')
        ax.set_title = (f'Freq: {f} Hz') 
        
    fig.suptitle(f'{meas_names[i]} \n Dinrange: {dinrange} dB', fontsize=14)

    
#%% DCISM Quadrature decomposition 
'''
        For those who are not initiated, This is the newly implemented 
    theory by Brandao. There is a lot going on here and a deeper theory review
    is required.
        Overall, pay attention to the imputs and keep the values that were set
    for num_gauss_pts, a, and b.
'''

num_gauss_pts = 25
a = 0 # lower limit of integral (always zero usually - we left it open for tests)
b = 30 # upper limit of integral (truncation)

list_gleg = []

for i, ppro_obj in enumerate(lista_ppro):
    decomp_qdt_gleg = Decomposition_QDT(p_mtx=lista_ppro[i].Hww_mtx, controls=controls,
        receivers=lista_ppro[i].meas_obj.receivers, source_coord=lista_ppro[i].meas_obj.source.coord[0],
        quad_order=num_gauss_pts, a = a, b = b, 
        retraction = 0, image_source_on = True, regu_par = 'gcv')

    #%
    list_gleg.append(decomp_qdt_gleg)  # Adiciona o objeto à lista

    # Choose sampling scheme
    decomp_qdt_gleg.gauss_legendre_sampling()
    
    # Solve
    decomp_qdt_gleg.pk_tikhonov(plot_l=False, method='Tikhonov')
    #decomp_qdt_gleg.least_squares_pk()
    
    # Reconstruct surface impedance
    decomp_qdt_gleg.zs(Lx=0.1, n_x=25, Ly=0.1, n_y=25, theta=[0], avgZs=True);

#%% Abs.Coef for all methods
'''
        This is the last part. We compare the different alphas obtained with 
    each method.
        I've commented lines inside the plot, but you can test different methods
    and see what it comes out. The real important ones are:
        - 2 mic method
        - NAH (plane wave expansion)
        - DCISM 
'''


for i, ppro_obj in enumerate(lista_ppro):
    plt.figure(figsize = (11,7))
    plt.semilogx(pwa_results[i][:,0], pwa_results[i][:,1], '-r', label = '2 mic', alpha = 0.7, linestyle=':',linewidth=2.5)
    plt.semilogx(list_zs_ded1[0].controls.freq, alpha_NAH[i], 'g', label = 'NAH', alpha = 0.7,linestyle='-',linewidth=2.5)
    plt.semilogx(list_zs_ded1[0].controls.freq, list_gleg[i].alpha.flatten(), 'b', label = 'DCISM', alpha = 0.7,linestyle='--',linewidth=2.5)
    # plt.semilogx(zs_ded.controls.freq, test, '-ob', label = 'Array -> vp_Surf', alpha = 0.4)
    # plt.semilogx(zs_ded.controls.freq, test2, '-or', label = 'Array -> alpha_from_pk', alpha = 0.4)
    # plt.semilogx(list_zs_ded1[0].controls.freq,list_zs_ded1[0].alpha.flatten(), 'k', label = 'NAH', alpha = 0.7)
        
    plt.legend(loc='lower right',fontsize=16)
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.ylim((-0.4, 1))
    plt.xlim((125, 4000))
    plt.xticks(ticks = [125,250,500,1000,2000,4000], labels = ['125','250','500','1000','2000','4000'], rotation = 0)
    plt.xlabel(r'Frequency [Hz]')
    plt.ylabel(r'$\alpha$ [-]')
    plt.yticks(fontsize=20)
    plt.xticks(ticks = [125,250,500,1000,2000,4000], labels = ['125','250','500','1000','2000','4000'], fontsize=20,rotation = 0)
    plt.xlabel(r'Frequency [Hz]', fontsize=22)
    plt.ylabel(r'$\alpha$ [-]', fontsize=22)
    # plt.title(f'{meas_names[i]} \n'"Elev. angle: {:.1f}$^\circ$".format(theta))
    plt.tight_layout();

#%% Quadro de PKs
import matplotlib.pyplot as plt
import numpy as np

freq = [600, 1500, 2500]
dinrange = 20

# row_labels = ['PU', 'PU Corrug', 'PU Mounted']
row_labels = ['PU','PU Corrug','PU Mounted']
nrows = len(list_zs_ded1)
ncols = len(freq)

subplot_size = 5  # tamanho quadrado de cada plot
fig_width = subplot_size * ncols + 3  # espaço extra à direita
fig_height = subplot_size * nrows 

fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
axes = np.atleast_2d(axes)

for i, zs_ded in enumerate(list_zs_ded1):
    for j, f in enumerate(freq):
        ax = axes[i, j]
        _, cbar = zs_ded.plot_ref_pkmap(
            ax=ax,
            freq=f,
            db=True,
            dinrange=dinrange,
            color_code='inferno'
        )

        if i == 0:
            ax.set_title(f'Freq: {f} Hz', fontsize=22)
        else:
            ax.set_title('')

        if j == 0:
            ax.set_ylabel(r'$k_y$ [rad/m]', fontsize=22)
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([])

        if j == ncols - 1:
            ax.annotate(row_labels[i],
                        xy=(1.05, 0.5), xycoords='axes fraction',
                        ha='left', va='center',
                        fontsize=22, fontweight='bold',
                        rotation=90)

        if i == nrows - 1:
            ax.set_xlabel(r'$k_x$ [rad/m]', fontsize=22)
        else:
            ax.set_xlabel('')
            ax.set_xticklabels([])

        ax.tick_params(axis='both', labelsize=16)

# Adiciona colorbar BEM mais à direita
cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])  # [left, bottom, width, height]

cbar_obj = fig.colorbar(cbar, cax=cbar_ax, ticks=np.linspace(-20, 0, 5))
cbar_obj.set_label(r'Magnitude [dB]', fontsize=20)  # ✅ aqui sim!

fig.colorbar(cbar, cax=cbar_ax, ticks=np.linspace(-20, 0, 5), label=r'Magnitude [dB]')

# Ajusta manualmente os espaçamentos entre subplots
plt.subplots_adjust(left=0.07, right=0.88, top=0.95, bottom=0.12, wspace=0.2, hspace=0.2)

fig.savefig('PU_ref_pkmap_matrix.pdf', dpi=400)
plt.show()

#%%    
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
import numpy as np

# Estilos por método
estilos_metodo = {
    'NAH': '--',
    'DCISM': '-'
}

# Nomes dos materiais
nomes_materiais = ["PU Corrug Mounted", "PU Corrug", "PU"]

# Cores distintas (evitando o amarelo claro)
cores = [cm.viridis(x) for x in [0.1, 0.5, 0.9]]  # Evita o meio (amarelado)

def lux(cor_rgb, fator):
    return np.clip(np.array(cor_rgb[:3]) * fator, 0, 1)

plt.figure(figsize=(11, 7))

for i, nome in enumerate(nomes_materiais):
    cor = cores[i]
    
    # NAH
    plt.semilogx(
        list_zs_ded1[0].controls.freq, alpha_NAH[i],
        linestyle=estilos_metodo['NAH'], linewidth=2.5,
        color=lux(cor, 1.0), alpha=0.8
    )

    # DCISM
    plt.semilogx(
        list_zs_ded1[0].controls.freq, list_gleg[i].alpha.flatten(),
        linestyle=estilos_metodo['DCISM'], linewidth=2.5,
        color=lux(cor, 0.7), alpha=0.8,
        label=nome  # Só a linha DCISM entra na legenda
    )

# Legenda principal com os materiais
plt.legend(loc='upper left', fontsize=10, title="Material")

# Legenda secundária para os métodos
linha_nah = mlines.Line2D([], [], color='gray', linestyle=estilos_metodo['NAH'], label='NAH', linewidth=2)
linha_dcism = mlines.Line2D([], [], color='gray', linestyle=estilos_metodo['DCISM'], label='DCISM', linewidth=2)

plt.gca().add_artist(plt.legend(handles=[linha_nah, linha_dcism], loc='upper left', title="Method", fontsize=10))

# Estética
plt.grid(True, which='both', linestyle='--', alpha=0.3)
plt.ylim((-0.2, 1))
plt.xlim((125, 4000))
plt.xlim((125, 4000))
plt.xticks(ticks = [125,250,500,1000,2000,4000], labels = ['125','250','500','1000','2000','4000'], rotation = 0)
plt.xlabel(r'Frequency [Hz]')
plt.ylabel(r'$\alpha$ [-]')
plt.title(f"PU absorption coefficient", fontsize=14)
plt.tight_layout()
plt.show()

#%%
import matplotlib.cm as cm
import matplotlib.lines as mlines
import numpy as np
import matplotlib.pyplot as plt

# Estilos por método
estilos_metodo = {
    'NAH': '--',
    'DCISM': '-'
}

# Nomes dos materiais
nomes_materiais = ["PU Corrug Mounted", "PU Corrug", "PU"]

# Cores distintas (evitando o amarelo claro)
cores = [cm.viridis(x) for x in [0.1, 0.5, 0.9]]

def lux(cor_rgb, fator):
    return np.clip(np.array(cor_rgb[:3]) * fator, 0, 1)

plt.figure(figsize=(12, 6))

# Armazenar os elementos para legenda
handles_legenda = []

for i, nome in enumerate(nomes_materiais):
    cor = cores[i]

    # NAH
    plt.semilogx(
        list_zs_ded1[0].controls.freq, alpha_NAH[i],
        linestyle=estilos_metodo['NAH'], linewidth=2.5,
        color=lux(cor, 1.0), alpha=0.8
    )
    handles_legenda.append(
        mlines.Line2D([], [], color=lux(cor, 1.0), linestyle=estilos_metodo['NAH'],
                      label=f"{nome} - NAH", linewidth=2.5)
    )

    # DCISM
    plt.semilogx(
        list_zs_ded1[0].controls.freq, list_gleg[i].alpha.flatten(),
        linestyle=estilos_metodo['DCISM'], linewidth=2.5,
        color=lux(cor, 0.7), alpha=0.8
    )
    handles_legenda.append(
        mlines.Line2D([], [], color=lux(cor, 0.7), linestyle=estilos_metodo['DCISM'],
                      label=f"{nome} - DCISM", linewidth=2.5)
    )

# Legenda combinada
plt.legend(handles=handles_legenda, loc='upper left', fontsize=16, title="Material + Method", ncol=1)

# Estética
plt.grid(True, which='both', linestyle='--', alpha=0.3)
plt.ylim((-0.21, 1))
plt.xlim((125, 4000))
plt.yticks(fontsize=20)
plt.xticks(ticks = [125,250,500,1000,2000,4000], labels = ['125','250','500','1000','2000','4000'], fontsize=20,rotation = 0)
plt.xlabel(r'Frequency [Hz]', fontsize=22)
plt.ylabel(r'$\alpha$ [-]', fontsize=22)
# plt.title(f"Coeficiente de absorção por material e método\nÂngulo de elevação: {theta:.1f}$^\circ$", fontsize=14)
plt.tight_layout()
plt.show()


#%%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter

# Dados
frequencias = list_zs_ded1[0].controls.freq
n_medicoes = len(lista_ppro)
n_freqs = len(frequencias)

# Matrizes (cada uma: n_medicoes x n_freqs)
alpha_2mic = np.array([pwa[:, 1] for pwa in pwa_results])
alpha_nah = np.array(alpha_NAH)
alpha_dcism = np.array([gleg.alpha.flatten() for gleg in list_gleg])

# Função para formatação dos ticks do eixo da barra de cores
def format_ticks_y(val, pos):
    return f'{val:.1f}'

# Criação do gráfico
fig, axes = plt.subplots(1, 3, figsize=(16, 9), sharey=True)

# Títulos para os métodos
metodos = ['2 Mic', 'PWE', 'DCISM']
matrizes = [alpha_2mic, alpha_nah, alpha_dcism]
cmaps = ['Reds', 'Greens', 'Blues']

for ax, alpha_matrix, metodo, cmap in zip(axes, matrizes, metodos, cmaps):
    c = ax.pcolormesh(frequencias, np.arange(1, n_medicoes + 1), alpha_matrix,
                      cmap=cmap, shading='auto', norm=mcolors.Normalize(vmin=-0.2, vmax=1.0))

    # Linhas horizontais tracejadas
    for i in range(1, n_medicoes):
        ax.hlines(i + 0.5, xmin=frequencias[0], xmax=frequencias[-1],
                  color='black', linestyle='--', linewidth=2)

    # Eixos
    ax.set_xscale('log')
    ax.set_xlim(125, 4000)
    ax.set_xticks([125, 250, 500, 1000, 2000, 4000])
    ax.set_xticklabels(['125', '250', '500', '1000', '2000', '4000'])
    ax.set_title(metodo, fontsize=16, fontweight='bold')

    # Barra de cores individual por subplot
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label(r'$\alpha$ [-]', fontsize=14, fontweight='bold')
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_ticks_y))

# Rótulos do eixo Y na lateral esquerda
axes[0].set_ylabel('Medições', fontsize=18, fontweight='bold')
axes[0].set_yticks(np.arange(1, n_medicoes + 1))
axes[0].set_yticklabels([f'Med {i + 1}' for i in range(n_medicoes)])

# Labels de X
for ax in axes:
    ax.set_xlabel('Frequência [Hz]', fontsize=16, fontweight='bold')

# Título geral
fig.suptitle(r'Coeficiente de absorção $\alpha$ por método', fontsize=20, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Salvar
fig.savefig('matriz_alpha_comparacao.png', dpi=300, bbox_inches='tight')
plt.show()

#%%

import matplotlib.cm as cm
import matplotlib.lines as mlines
import numpy as np
import matplotlib.pyplot as plt

# Estilos por método
estilos_metodo = {
    'PWE': '--',
    'DCISM': '-',
    '2 Mics': ':'  # Novo estilo para 2 microfones
}

# Nomes dos materiais
nomes_materiais = ["PU Corrug Mounted", "PU Corrug", "PU"]

# Cores distintas (evitando o amarelo claro)
cores = [cm.viridis(x) for x in [0.1, 0.5, 0.9]]

def lux(cor_rgb, fator):
    return np.clip(np.array(cor_rgb[:3]) * fator, 0, 1)

plt.figure(figsize=(11, 6))

# Armazenar os elementos para legenda
handles_legenda = []

for i, nome in enumerate(nomes_materiais):
    cor = cores[i]

    # NAH
    plt.semilogx(
        list_zs_ded1[0].controls.freq, alpha_NAH[i],
        linestyle=estilos_metodo['PWE'], linewidth=2.5,
        color=lux(cor, 1.0), alpha=0.8
    )
    handles_legenda.append(
        mlines.Line2D([], [], color=lux(cor, 1.0), linestyle=estilos_metodo['PWE'],
                      label=f"{nome} - PWE", linewidth=2.5,alpha = 0.6)
    )

    # DCISM
    plt.semilogx(
        list_zs_ded1[0].controls.freq, list_gleg[i].alpha.flatten(),
        linestyle=estilos_metodo['DCISM'], linewidth=2.5,
        color=lux(cor, 0.7), alpha=0.6
    )
    handles_legenda.append(
        mlines.Line2D([], [], color=lux(cor, 0.7), linestyle=estilos_metodo['DCISM'],
                      label=f"{nome} - DCISM", linewidth=2.5)
    )

    # 2 Mics (PWA)
    plt.semilogx(
        pwa_results[i][:, 0], pwa_results[i][:, 1],
        linestyle=estilos_metodo['2 Mics'], linewidth=2.5,
        color=lux(cor, 0.4), alpha=1
    )
    handles_legenda.append(
        mlines.Line2D([], [], color=lux(cor, 0.5), linestyle=estilos_metodo['2 Mics'],
                      label=f"{nome} - 2 Mics", linewidth=3.5)
    )

# Legenda combinada
plt.legend(handles=handles_legenda, loc='upper left', fontsize=14, ncol=1)

# Estética
plt.grid(True, which='both', linestyle='--', alpha=0.3)
plt.ylim((-0.21, 1))
plt.xlim((125, 4000))
plt.yticks(fontsize=20)
plt.xticks(ticks=[125, 250, 500, 1000, 2000, 4000],
           labels=['125', '250', '500', '1000', '2000', '4000'],
           fontsize=20, rotation=0)
plt.xlabel(r'Frequency [Hz]', fontsize=22)
plt.ylabel(r'$\alpha$ [-]', fontsize=22)
# plt.title(f"Coeficiente de absorção por material e método\nÂngulo de elevação: {theta:.1f}$^\circ$", fontsize=14)
plt.tight_layout()
plt.show()

#%% cores iguais

# Estilos por método
estilos_metodo = {
    'NAH': '--',
    'DCISM': '-',
    '2 Mics': ':'
}

# Nomes dos materiais
nomes_materiais = ["PU", "PU Corrug", "PU Corrug Mounted"]

# Cores (iguais entre métodos, uma por material)
cores = [cm.viridis(x) for x in [0.1, 0.5, 0.9]]

def lux(cor_rgb, fator):
    return np.clip(np.array(cor_rgb[:3]) * fator, 0, 1)

plt.figure(figsize=(11, 7))

# Armazenar os elementos para legenda
handles_legenda = []

for i, nome in enumerate(nomes_materiais):
    cor = cores[i]  # cor base para o material

    # NAH
    plt.semilogx(
        list_zs_ded1[0].controls.freq, alpha_NAH[i],
        linestyle=estilos_metodo['NAH'], linewidth=2.5,
        color=cor, alpha=0.4
    )
    handles_legenda.append(
        mlines.Line2D([], [], color=cor, linestyle=estilos_metodo['NAH'],
                      label=f"{nome} - NAH", linewidth=2.5, alpha=0.6)
    )

    # DCISM
    plt.semilogx(
        list_zs_ded1[0].controls.freq, list_gleg[i].alpha.flatten(),
        linestyle=estilos_metodo['DCISM'], linewidth=2.5,
        color=cor, alpha=0.6
    )
    handles_legenda.append(
        mlines.Line2D([], [], color=cor, linestyle=estilos_metodo['DCISM'],
                      label=f"{nome} - DCISM", linewidth=2.5, alpha=0.8)
    )

    # 2 Mics (mais destaque)
    plt.semilogx(
        pwa_results[i][:, 0], pwa_results[i][:, 1],
        linestyle=estilos_metodo['2 Mics'], linewidth=3,
        color=cor, alpha=1.0
    )
    handles_legenda.append(
        mlines.Line2D([], [], color=cor, linestyle=estilos_metodo['2 Mics'],
                      label=f"{nome} - 2 Mics", linewidth=2.5, alpha=1.0)
    )

# Legenda combinada
plt.legend(handles=handles_legenda, loc='upper left', fontsize=15, ncol=1)

# Estética
plt.grid(True, which='both', linestyle='--', alpha=0.3)
plt.ylim((-0.21, 1))
plt.xlim((125, 4000))
plt.yticks(fontsize=20)
plt.xticks(ticks=[125, 250, 500, 1000, 2000, 4000],
           labels=['125', '250', '500', '1000', '2000', '4000'],
           fontsize=20, rotation=0)
plt.xlabel(r'Frequency [Hz]', fontsize=22)
plt.ylabel(r'$\alpha$ [-]', fontsize=22)
# plt.title(f"Coeficiente de absorção por material e método\nÂngulo de elevação: {theta:.1f}$^\circ$", fontsize=14)
plt.tight_layout()
plt.show()

