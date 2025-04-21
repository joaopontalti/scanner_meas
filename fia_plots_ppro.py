    # -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 09:27:39 2024

@author: joaop
"""

##############################################################################
#### Plotagem dos gráficos de interesse para artigo FIA 2024 ####

import matplotlib.pyplot as plt
import os
import numpy as np
from controlsair import AirProperties, AlgControls#, add_noise, add_noise2
from sources import Source
from receivers import Receiver
from qterm_estimation import ImpedanceDeductionQterm
from ppro_meas_insitu import InsituMeasurementPostPro
plt.rcParams.update({'font.size': 22})  # Ajuste o valor conforme necessário
import pytta
import matplotlib.ticker as ticker 
from matplotlib.ticker import FuncFormatter # formata a quantidade de casas após a virgula
import locale #substituir ponto por virgula
from joaop_functions import sph2cart, format_meas_name, select_mics, list_folders,meas_info

#%%
main_folder = "C:/Users/joaop/anaconda3/envs/Dataset_JoaoP/TCC_DATASET_JoaoP/DATA2025/" 
keyword = "2planar"

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
    # ppro_obj.compute_all_ir_load(regularization = True,  deconv_with_rec = True, 
                    # only_linear_part = True) 
    
    ppro_obj.load_irs()
#%% Precisao dos plots
# Função para formatar os ticks com ponto ao invés de vírgula
def format_ticks_x(x, pos):
    return f'{x:.3f}'.replace('.', ',')
def format_ticks_y(y, pos):
    return f'{y:.1f}'.replace('.', ',')

#%% Nome dos arquivos .pkl sendo importados

main_folder = 'C:/Users/Gus/anaconda3/envs/Gus-env/repo_github/gus_meas'
source = Source([0, 0, 0.3])

# name1 = 'melamina_L60cm_d3cm_s100cm_2mics_19092024_0'

name1 = 'PET_L120cm_L60cm_d4cm_s30cm_2mics_25102024_1' #'PET_grooved_plate' # 'melamine' #
name2 = 'PET_L120cm_L60cm_d4cm_s30cm_2mics_25102024_2' #'PET_grooved_plate' # 'melamine' #
name3 = 'PET_L120cm_L60cm_d4cm_s30cm_2mics_25102024_3' #'PET_grooved_plate' # 'melamine' #
name4 = 'PET_L120cm_L60cm_d4cm_s30cm_2mics_25102024_15' #'PET_grooved_plate' # 'melamine' #
name5 = 'PET_L120cm_L60cm_d4cm_s30cm_2mics_25102024_5' #'PET_grooved_plate' # 'melamine' #
name6 = 'PET_L120cm_L60cm_d4cm_s30cm_2mics_25102024_6' #'PET_grooved_plate' # 'melamine' #
name7 = 'PET_L120cm_L60cm_d4cm_s30cm_2mics_25102024_7' #'PET_grooved_plate' # 'melamine' #
name8 = 'PET_L120cm_L60cm_d4cm_s30cm_2mics_25102024_8' #'PET_grooved_plate' # 'melamine' #
name9 = 'PET_L120cm_L60cm_d4cm_s30cm_2mics_25102024_9' #'PET_grooved_plate' # 'melamine' #
name10= 'PET_L120cm_L60cm_d4cm_s30cm_2mics_25102024_10' #'PET_grooved_plate' # 'melamine' #

#%% Instanciando o objeto de pos processamento - carrega o meas_obj

####### Considerar ideia p futuro: pre listar os nomes com um leitor de folder
    # que permite ao usuário escolher as pastas a partir do console

ppro1 = lista_ppro[0]

ppro2 = lista_ppro[1]
ppro3 = lista_ppro[2]

# ppro2 = InsituMeasurementPostPro(main_folder = main_folder, name = name2)
# ppro2.meas_obj.source = source

# ppro3 = InsituMeasurementPostPro(main_folder = main_folder, name = name3)
# ppro3.meas_obj.source = source

# ppro4 = InsituMeasurementPostPro(main_folder = main_folder, name = name4)
# ppro4.meas_obj.source = source

# ppro5 = InsituMeasurementPostPro(main_folder = main_folder, name = name5)
# ppro5.meas_obj.source = source

# ppro6 = InsituMeasurementPostPro(main_folder = main_folder, name = name6)
# ppro6.meas_obj.source = source

# ppro7 = InsituMeasurementPostPro(main_folder = main_folder, name = name7)
# ppro7.meas_obj.source = source

# ppro8 = InsituMeasurementPostPro(main_folder = main_folder, name = name8)
# ppro8.meas_obj.source = source

# ppro9 = InsituMeasurementPostPro(main_folder = main_folder, name = name9)
# ppro9.meas_obj.source = source

# ppro10 = InsituMeasurementPostPro(main_folder = main_folder, name = name10)
# ppro10.meas_obj.source = source

#%% Computando as RIs e aplicando regularização
ppro1.compute_all_ir_load(regularization = True, only_linear_part = True)
ppro2.compute_all_ir_load(regularization = True, only_linear_part = True)
ppro3.compute_all_ir_load(regularization = True, only_linear_part = True)
# ppro4.compute_all_ir_load(regularization = True, only_linear_part = True)
# ppro5.compute_all_ir_load(regularization = True, only_linear_part = True)
# ppro6.compute_all_ir_load(regularization = True, only_linear_part = True)
# ppro7.compute_all_ir_load(regularization = True, only_linear_part = True)
# ppro8.compute_all_ir_load(regularization = True, only_linear_part = True)
# ppro9.compute_all_ir_load(regularization = True, only_linear_part = True)
# ppro10.compute_all_ir_load(regularization = True, only_linear_part = True)


#%% Carregando as RIs e computando o espectro
ppro1.load_irs()
ppro2.load_irs()
ppro3.load_irs()
# ppro4.load_irs()
# ppro5.load_irs()
# ppro6.load_irs()
# ppro7.load_irs()
# ppro8.load_irs()
# ppro9.load_irs()
# ppro10.load_irs()
ppro1.compute_spk()
ppro2.compute_spk()
ppro3.compute_spk()
# ppro4.compute_spk()
# ppro5.compute_spk()
# ppro6.compute_spk()
# ppro7.compute_spk()
# ppro8.compute_spk()
# ppro9.compute_spk()
# ppro10.compute_spk()

#%% Janela Adrienne para cada tipo de medição
#melamina
# tstart = 0e-3
# tend = 12.5e-3
# tfim = 20e-3
# dt_fadein = 0.1e-3
# dt_fadeout = 1e-3

#PET
tstart = 0e-3
tend = 13.7e-3
tfim = 20e-3
dt_fadein = 0.1e-3
dt_fadeout = 1e-3

ppro1.set_adrienne_win(tstart = tstart, dt_fadein = dt_fadein, t_cutoff = tend, dt_fadeout = dt_fadeout)
ppro1.apply_window()

ppro2.set_adrienne_win(tstart = tstart, dt_fadein = dt_fadein, t_cutoff = tend, dt_fadeout = dt_fadeout)
ppro2.apply_window()

ppro3.set_adrienne_win(tstart = tstart, dt_fadein = dt_fadein, t_cutoff = tend, dt_fadeout = dt_fadeout)
ppro3.apply_window()

# ppro4.set_adrienne_win(tstart = tstart, dt_fadein = dt_fadein, t_cutoff = tend, dt_fadeout = dt_fadeout)
# ppro4.apply_window()

# ppro5.set_adrienne_win(tstart = tstart, dt_fadein = dt_fadein, t_cutoff = tend, dt_fadeout = dt_fadeout)
# ppro5.apply_window()

# ppro6.set_adrienne_win(tstart = tstart, dt_fadein = dt_fadein, t_cutoff = tend, dt_fadeout = dt_fadeout)
# ppro6.apply_window()

# ppro7.set_adrienne_win(tstart = tstart, dt_fadein = dt_fadein, t_cutoff = tend, dt_fadeout = dt_fadeout)
# ppro7.apply_window()

# ppro8.set_adrienne_win(tstart = tstart, dt_fadein = dt_fadein, t_cutoff = tend, dt_fadeout = dt_fadeout)
# ppro8.apply_window()

# ppro9.set_adrienne_win(tstart = tstart, dt_fadein = dt_fadein, t_cutoff = tend, dt_fadeout = dt_fadeout)
# ppro9.apply_window()

# ppro10.set_adrienne_win(tstart = tstart, dt_fadein = dt_fadein, t_cutoff = tend, dt_fadeout = dt_fadeout)
# ppro10.apply_window()

##############################################################################################
################################################################################

############################# PLOTS ############################################

#%% Plote o gráfico de janelamento para várias RIs

def plot_ir_comparison(ppro1, rec_index, xlims=(0, 50e-3), normalize=True):
    """ Compara IR antes e depois da aplicação da janela """
    
    if normalize:
        ht = ppro1.ht_mtx[rec_index, :] / np.amax(ppro1.ht_mtx[rec_index, :])
        htw = ppro1.htw_mtx[rec_index, :] / np.amax(ppro1.htw_mtx[rec_index, :])
    else:
        ht = ppro1.ht_mtx[rec_index, :]
        htw = ppro1.htw_mtx[rec_index, :]
        
    # # Plots em inglês
    # plt.figure(figsize=(12, 12))  # Define o tamanho da figura
    # plt.plot(ppro1.time_ht, ht, '-k', label='IR raw', linewidth=1.8)
    # plt.plot(ppro1.time_ht, htw, 'red', label='IR windowed', alpha=0.5,linewidth=2.8)
    # plt.plot(ppro1.time_ht, ppro1.adrienne_win, '--b', label='Adrienne window', alpha=0.7, linewidth=2)
    # plt.grid(True)
    # plt.legend(loc='upper right',fontsize=20)
    # plt.xlim(xlims)
    # plt.xlabel(r'Time [s]', fontsize=22, fontweight='bold')
    # plt.ylabel(r'Amplitude [-]', fontsize=22, fontweight='bold')
    
    
    # Plots em português
    plt.figure(figsize=(12, 12))  # Define o tamanho da figura
    plt.plot(ppro1.time_ht, ht, '-k', label='IR raw', linewidth=1.8)
    plt.plot(ppro1.time_ht, htw, 'red', label='IR windowed', alpha=0.5,linewidth=2.8)
    plt.plot(ppro1.time_ht, ppro1.adrienne_win, '--b', label='Adrienne Window', alpha=0.7, linewidth=2)
    plt.grid(True,alpha=0.3)
    plt.legend(loc='upper right',fontsize=20)
    plt.xlim(xlims)
    plt.xlabel(r'Time [s]', fontsize=22, fontweight='bold')
    plt.ylabel(r'Amplitude [-]', fontsize=22, fontweight='bold')


    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(fontsize=24)

    def format_ticks_x(x, pos):
        return f'{x:.1f}'.replace('.', ',')

    # Aplicar o formatador personalizado aos eixos x e y
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_ticks_x))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_ticks_y))

    plt.show()    
#%% Plot de 1 IR
    # plt.plot(ppro1.time_ht * 1000, ppro1.time_ht_mtx[0, :] / np.amax(ppro1.ht_mtx[0, :]), label = 'IR')
    plt.plot(lista_ppro[i].time_ht * 1000, lista_ppro[i].ht_mtx[0, :] / np.amax(lista_ppro[i].ht_mtx[0, :]), label = 'IR')

    # Ajuste dos ticks do eixo x e y, se necessário
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.tight_layout()  # Ajusta o layout para evitar sobreposição

    plt.show()

    # def format_ticks_y(x,pos):
    #     return f'{x:.1f}'.replace('.',',')
    # def format_ticks_x(x, pos):
    #     return f'{x:.3f}'.replace('.', ',')

rec_index = 1
xlims = (0, 25e-3)

plot_ir_comparison(lista_ppro[0], rec_index, xlims=xlims)

rec_index = 1;
tlims = (0e-3, 25e-3);

point_x = 15e-3  # Coordenada x para a seta
point_y = 0.02   # Coordenada y para a seta


plt.annotate('First room reflections',  # Texto da caixa
              xy=(point_x, point_y),  # Ponto para onde a seta aponta
              xytext=(point_x + 0.000, point_y + 0.511),  # Localização da caixa de texto
              textcoords='data',
              bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='lightyellow'),
              arrowprops=dict(facecolor='black', arrowstyle='->', lw=2),
              fontsize = 25)

plt.annotate('Region of interest',  # Texto da caixa
              xy=(14e-3, 7e-3),  # Ponto para onde a seta aponta
              xytext=(3.5e-3, 5e-3 + 0.57),  # Localização da caixa de texto
              textcoords='data',
              bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='lightyellow'),
              fontsize = 25)


# Aplicar o formatador personalizado aos eixos x e y
plt.gca().xaxis.set_major_formatter(FuncFormatter(format_ticks_x))
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_ticks_y))

#%% Plot das RIs com setas

tlims = (0e-3 * 1000, 5e-3 * 1000)  # Ajustando os limites para milissegundos
fig, ax = plt.subplots(1, figsize=(16, 14), sharex=False)

# Plotando a IR do mic 1 sem normalização
# ht1 = ppro1.htw_mtx[0, :]
ht1 = ppro1.ht_mtx[0, :]  # RI do mic 1
ht1 = ht1 / np.amax(ht1)     # normalizando o vetor da RI
time1 = ppro1.time_ht * 1000  # Convertendo o vetor temporal para milissegundos
time2 = ppro1.time_ht * 1000  # Convertendo o vetor temporal para milissegundos
ax.plot(time1, ht1, label="RI Mic 1 - (0, 0, 0,02) m",color='red',linewidth=3.5)
max_idx1 = np.argmax(ht1)
max_idx2 = np.argmax(ht1)
max_time1 = time1[max_idx1]

formatted_max_time1 = f'{max_time1:.2f}'.replace('.', ',')
# formatted_max_time2 = f'{max_time2:.2f}'.replace('.', ',')

ax.annotate(f'{formatted_max_time1} ms', 
            xy=(max_time1, ht1[max_idx1]), 
            xytext=(max_time1, ht1[max_idx1] + 0.12),  # Ajuste a posição do texto
            arrowprops=dict(facecolor='red', shrink=0.05, width=6.5),
            rotation=45,fontsize=21,fontweight='normal')  # Inclina o texto em 45 graus
            
#ax.axvline(max_time1, color='red', linestyle='--', linewidth=2,label=f'{max_time1:.2f} ms')

# Plotando a IR do mic 2 sem normalização
ht2 = ppro1.ht_mtx[1, :]  # RI do mic 2
ht2 = ht2 / np.amax(ht2)     # normalizando o vetor da RI
ax.plot(time1, ht2, label="RI Mic 2 - (0, 0, 0,04) m",color='black',linewidth=3.5)
max_idx2 = np.argmax(ht2)
max_time2 = time1[max_idx2]
formatted_max_time2 = f'{max_time2:.2f}'.replace('.', ',')

ax.annotate(f'{formatted_max_time2} ms', 
            xy=(max_time2, ht2[max_idx2]), 
            xytext=(max_time2, ht2[max_idx2] + 0.12),  # Ajuste a posição do texto
            arrowprops=dict(facecolor='black', shrink=0.05, width=6.5),
            rotation=45,fontsize=21,fontweight='normal')  # Inclina o texto em 45 graus
              
#ax.axvline(max_time2, color='black', linestyle='--', linewidth=2, label=f'{max_time2:.2f} ms')

# Ajustes finais no gráfico
ax.grid()
ax.set_xlim(tlims)  # Definindo limites do eixo X em milissegundos
ax.set_ylim([-1.0, 1.5])  # Definindo limites do eixo Y (ajuste conforme necessário)
plt.xlabel('Tempo [ms]', fontsize=26, fontweight='bold')
plt.ylabel('Amplitude [-]', fontsize=26, fontweight='bold')
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.legend(fontsize=24)

def format_ticks_x(x, pos):
    return f'{x:.1f}'.replace('.', ',')

# Aplicar o formatador personalizado aos eixos x e y
plt.gca().xaxis.set_major_formatter(FuncFormatter(format_ticks_x))
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_ticks_y))

plt.show()

#%% Plot das RIs juntas

def format_ticks_x(x, pos):
    return f'{x:.3f}'.replace('.', ',')
def format_ticks_y(y, pos):
    return f'{y:.1f}'.replace('.', ',')

# Configurações dos eixos
flims = (0, 4000)
ax.set_xlim(flims)
ax.set_ylim((-80, 0))
ax.set_xlabel('Frequência [Hz]',fontsize=22,fontweight='bold')
ax.set_ylabel('Amplitude [-]',fontsize=22,fontweight='bold')
ax.legend()

######### PLOT COM CONFIGURAÇAO DOS DEGRADES ########################
# plt.figure(figsize=(12, 9))
# plt.plot(ppro1.time_ht, ppro1.ht_mtx[0, :] / np.amax(ppro1.ht_mtx[0, :]), label='med 1', linewidth=16.5, color='brown', alpha=1)
# plt.plot(ppro2.time_ht, ppro2.ht_mtx[0, :] / np.amax(ppro2.ht_mtx[0, :]), label='med 2', linewidth=15.7, color='magenta', alpha=0.2)
# plt.plot(ppro3.time_ht, ppro3.ht_mtx[0, :] / np.amax(ppro3.ht_mtx[0, :]), label='med 3', linewidth=14.5, color='purple', alpha=1)
# plt.plot(ppro4.time_ht, ppro4.ht_mtx[0, :] / np.amax(ppro4.ht_mtx[0, :]), label='med 4', linewidth=13.6, color='blue', alpha=0.2)
# plt.plot(ppro5.time_ht, ppro5.ht_mtx[0, :] / np.amax(ppro5.ht_mtx[0, :]), label='med 5', linewidth=11.8, color='cyan', alpha=1)
# plt.plot(ppro6.time_ht, ppro6.ht_mtx[0, :] / np.amax(ppro6.ht_mtx[0, :]), label='med 6', linewidth=10, color='green', alpha=0.2)
# plt.plot(ppro7.time_ht, ppro7.ht_mtx[0, :] / np.amax(ppro7.ht_mtx[0, :]), label='med 7', linewidth=7.6, color='lime', alpha=1)
# plt.plot(ppro8.time_ht, ppro8.ht_mtx[0, :] / np.amax(ppro8.ht_mtx[0, :]), label='med 8', linewidth=6.3, color='yellow', alpha=0.2)
# plt.plot(ppro9.time_ht, ppro9.ht_mtx[0, :] / np.amax(ppro9.ht_mtx[0, :]), label='med 9', linewidth=5, color='orange', alpha=1)
# plt.plot(ppro10.time_ht, ppro10.ht_mtx[0, :] / np.amax(ppro10.ht_mtx[0, :]), label='med 10', linewidth=10, color='red', alpha=0.2)

# plt.figure(figsize=(12, 9))
# plt.plot(ppro1.time_ht, ppro1.ht_mtx[0, :] / np.amax(ppro1.ht_mtx[0, :]), label='med 1', linewidth=4, color='brown', alpha=0.5)
# plt.plot(ppro2.time_ht, ppro2.ht_mtx[0, :] / np.amax(ppro2.ht_mtx[0, :]), label='med 2', linewidth=4, color='magenta', alpha=0.5)
# plt.plot(ppro3.time_ht, ppro3.ht_mtx[0, :] / np.amax(ppro3.ht_mtx[0, :]), label='med 3', linewidth=4, color='purple', alpha=0.5)
# plt.plot(ppro4.time_ht, ppro4.ht_mtx[0, :] / np.amax(ppro4.ht_mtx[0, :]), label='med 4', linewidth=4, color='blue', alpha=0.5)
# plt.plot(ppro5.time_ht, ppro5.ht_mtx[0, :] / np.amax(ppro5.ht_mtx[0, :]), label='med 5', linewidth=4, color='cyan', alpha=0.5)
# plt.plot(ppro6.time_ht, ppro6.ht_mtx[0, :] / np.amax(ppro6.ht_mtx[0, :]), label='med 6', linewidth=4, color='green', alpha=0.5)
# plt.plot(ppro7.time_ht, ppro7.ht_mtx[0, :] / np.amax(ppro7.ht_mtx[0, :]), label='med 7', linewidth=4, color='lime', alpha=0.5)
# plt.plot(ppro8.time_ht, ppro8.ht_mtx[0, :] / np.amax(ppro8.ht_mtx[0, :]), label='med 8', linewidth=4, color='yellow', alpha=0.5)
# plt.plot(ppro9.time_ht, ppro9.ht_mtx[0, :] / np.amax(ppro9.ht_mtx[0, :]), label='med 9', linewidth=4, color='orange', alpha=0.5)
# plt.plot(ppro10.time_ht, ppro10.ht_mtx[0, :] / np.amax(ppro10.ht_mtx[0, :]), label='med 10', linewidth=4, color='red', alpha=0.5)


plt.figure(figsize=(12, 9))
# plt.plot(ppro1.time_ht, ppro1.ht_mtx[0, :] / np.amax(ppro1.ht_mtx[0, :]), label='med 1', linewidth=17.5, color='brown')
# plt.plot(ppro2.time_ht, ppro2.ht_mtx[0, :] / np.amax(ppro2.ht_mtx[0, :]), label='med 2', linewidth=15.5, color='magenta')
# plt.plot(ppro3.time_ht, ppro3.ht_mtx[0, :] / np.amax(ppro3.ht_mtx[0, :]), label='med 3', linewidth=13.5, color='purple')
# plt.plot(ppro4.time_ht, ppro4.ht_mtx[0, :] / np.amax(ppro4.ht_mtx[0, :]), label='med 4', linewidth=11.5, color='blue')
# plt.plot(ppro5.time_ht, ppro5.ht_mtx[0, :] / np.amax(ppro5.ht_mtx[0, :]), label='med 5', linewidth=10.5, color='cyan')
# plt.plot(ppro6.time_ht, ppro6.ht_mtx[0, :] / np.amax(ppro6.ht_mtx[0, :]), label='med 6', linewidth=8.5, color='green')
# plt.plot(ppro7.time_ht, ppro7.ht_mtx[0, :] / np.amax(ppro7.ht_mtx[0, :]), label='med 7', linewidth=7.5, color='lime')
# plt.plot(ppro8.time_ht, ppro8.ht_mtx[0, :] / np.amax(ppro8.ht_mtx[0, :]), label='med 8', linewidth=5.5, color='yellow')
# plt.plot(ppro9.time_ht, ppro9.ht_mtx[0, :] / np.amax(ppro9.ht_mtx[0, :]), label='med 9', linewidth=3.5, color='black')
# plt.plot(ppro10.time_ht, ppro10.ht_mtx[0, :] / np.amax(ppro10.ht_mtx[0, :]), label='med 10', linewidth=.5, color='red')

plt.plot(ppro1.time_ht, ppro1.ht_mtx[0, :] / np.amax(ppro1.ht_mtx[0, :]), label='Med. 01', linewidth=2, color='brown')
plt.plot(ppro2.time_ht, ppro2.ht_mtx[0, :] / np.amax(ppro2.ht_mtx[0, :]), label='Med. 02', linewidth=2, color='magenta')
plt.plot(ppro3.time_ht, ppro1.ht_mtx[0, :] / np.amax(ppro1.ht_mtx[0, :]), label='Med. 03*', linewidth=2, color='purple')
plt.plot(ppro4.time_ht, ppro4.ht_mtx[0, :] / np.amax(ppro4.ht_mtx[0, :]), label='Med. 04', linewidth=2, linestyle = '--', color='blue')
plt.plot(ppro5.time_ht, ppro5.ht_mtx[0, :] / np.amax(ppro5.ht_mtx[0, :]), label='Med. 05', linewidth=2, linestyle = '--', color='cyan')
plt.plot(ppro6.time_ht, ppro6.ht_mtx[0, :] / np.amax(ppro6.ht_mtx[0, :]), label='Med. 06', linewidth=2, linestyle = '--', color='green')
plt.plot(ppro7.time_ht, ppro7.ht_mtx[0, :] / np.amax(ppro7.ht_mtx[0, :]), label='Med. 07', linewidth=2, linestyle = '-', color='lime')
plt.plot(ppro8.time_ht, ppro8.ht_mtx[0, :] / np.amax(ppro8.ht_mtx[0, :]), label='Med. 08', linewidth=2, linestyle = '-', color='yellow')
plt.plot(ppro9.time_ht, ppro9.ht_mtx[0, :] / np.amax(ppro9.ht_mtx[0, :]), label='Med. 09', linewidth=2, linestyle = '--', color='black')
plt.plot(ppro10.time_ht, ppro10.ht_mtx[0, :] / np.amax(ppro10.ht_mtx[0, :]), label='Med. 10', linewidth=1, color='red')


# Configuração do intervalo de tempo
plt.xlim(0, tfim)
plt.grid()
plt.legend(loc = 'upper right')
plt.xlabel(r'Tempo [s]', fontsize=22, fontweight='bold')
plt.ylabel(r'Amplitude [-]', fontsize=22, fontweight='bold')

# Aplicar o formatador personalizado aos eixos x e y
plt.gca().xaxis.set_major_formatter(FuncFormatter(format_ticks_x))
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_ticks_y))

# plt.savefig("IRs_normalizadas_29072024.pdf", format="pdf")
plt.show()
#%%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter

# Definindo funções para formatação dos ticks
def format_ticks_ms(x, pos):
    return f'{x:.1f}'.replace('.', ',')

def format_ticks_x(x, pos):
    return f'{x:.1f}'

def format_ticks_y(y, pos):
    return f'{y:.1f}'.replace('.', ',')


#%% Criando os subplots
fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=(18, 10))

# Definindo estilos de linha e largura para os gráficos
line_styles = {0: ':', 1: '-', 2: '-.', 3: '--', 4: '-'}
line_widths = {0: 13.5, 1: 5.5, 2: 8.8, 3: 4.5, 4: 1.7}

# Primeiro subplot (esquerda - Medições 1 a 5 no tempo)
ax1.plot(ppro1.time_ht * 1000, ppro1.ht_mtx[0, :] / np.amax(ppro1.ht_mtx[0, :]), label='IR med 1', linewidth=line_widths[0], color='#FF0000', linestyle=line_styles[0])
ax1.plot(ppro2.time_ht * 1000, ppro2.ht_mtx[0, :] / np.amax(ppro2.ht_mtx[0, :]), label='IR med 2', linewidth=line_widths[1], color='#0000FF', linestyle=line_styles[1])
ax1.plot(ppro3.time_ht * 1000, ppro3.ht_mtx[0, :] / np.amax(ppro3.ht_mtx[0, :]), label='IR med 3', linewidth=line_widths[2], color='#FFFF00', linestyle=line_styles[2])
ax1.plot(ppro4.time_ht * 1000, ppro4.ht_mtx[0, :] / np.amax(ppro4.ht_mtx[0, :]), label='IR med 4', linewidth=line_widths[3], color='#00FF00', linestyle=line_styles[3])
ax1.plot(ppro5.time_ht * 1000, ppro5.ht_mtx[0, :] / np.amax(ppro5.ht_mtx[0, :]), label='IR med 5', linewidth=line_widths[4], color='#800080', linestyle=line_styles[4])

# Configurações do primeiro subplot (tempo)
ax1.set_xlim(0, tfim * 1000)
ax1.grid(True)
ax1.legend(loc='upper right', fontsize=17)  # Legenda no canto inferior direito
ax1.set_ylabel(r'Amplitude [-]', fontsize=22, fontweight='bold')
ax1.xaxis.set_major_formatter(FuncFormatter(format_ticks_ms))
ax1.yaxis.set_major_formatter(FuncFormatter(format_ticks_y))
ax1.tick_params(axis='x', labelbottom=False)

# Segundo subplot (esquerda - Medições 6 a 10 no tempo)
ax2.plot(ppro6.time_ht * 1000, ppro6.ht_mtx[0, :] / np.amax(ppro6.ht_mtx[0, :]), label='IR med 6', linewidth=line_widths[0], color='#00FFFF', linestyle=line_styles[0])
ax2.plot(ppro7.time_ht * 1000, ppro7.ht_mtx[0, :] / np.amax(ppro7.ht_mtx[0, :]), label='IR med 7', linewidth=line_widths[1], color='#FFA500', linestyle=line_styles[1])
ax2.plot(ppro8.time_ht * 1000, ppro8.ht_mtx[0, :] / np.amax(ppro8.ht_mtx[0, :]), label='IR med 8', linewidth=line_widths[2], color='#FF00FF', linestyle=line_styles[2])
ax2.plot(ppro9.time_ht * 1000, ppro9.ht_mtx[0, :] / np.amax(ppro9.ht_mtx[0, :]), label='IR med 9', linewidth=line_widths[3], color='#008000', linestyle=line_styles[3])
ax2.plot(ppro10.time_ht * 1000, ppro10.ht_mtx[0, :] / np.amax(ppro10.ht_mtx[0, :]), label='IR med 10', linewidth=line_widths[4], color='#000000', linestyle=line_styles[4])

# Configurações do segundo subplot (tempo)
ax2.set_xlim(0, tfim * 1000)
ax2.grid(True)
ax2.legend(loc='upper right', fontsize=17)  # Legenda no canto inferior direito
ax2.set_xlabel(r'Time [ms]', fontsize=28, fontweight='bold')
ax2.set_ylabel(r'Amplitude [-]', fontsize=22, fontweight='bold')
ax2.xaxis.set_major_formatter(FuncFormatter(format_ticks_ms))
ax2.yaxis.set_major_formatter(FuncFormatter(format_ticks_y))

# Terceiro subplot (direita - Medições 1 a 5 na frequência)
ticks = [125, 250, 500, 1000, 2000, 4000]
flims = (100, 4000)

ax3.semilogx(ppro1.freq_Hw, 20 * np.log10(np.abs(ppro1.Hww_mtx[0, :])), label='FRF med 1', linewidth=line_widths[0], color='#FF0000', linestyle=line_styles[0])
ax3.semilogx(ppro2.freq_Hw, 20 * np.log10(np.abs(ppro2.Hww_mtx[0, :])), label='FRF med 2', linewidth=line_widths[1], color='#0000FF', linestyle=line_styles[1])
ax3.semilogx(ppro3.freq_Hw, 20 * np.log10(np.abs(ppro3.Hww_mtx[0, :])), label='FRF med 3', linewidth=line_widths[2], color='#FFFF00', linestyle=line_styles[2])
ax3.semilogx(ppro4.freq_Hw, 20 * np.log10(np.abs(ppro4.Hww_mtx[0, :])), label='FRF med 4', linewidth=line_widths[3], color='#00FF00', linestyle=line_styles[3])
ax3.semilogx(ppro5.freq_Hw, 20 * np.log10(np.abs(ppro5.Hww_mtx[0, :])), label='FRF med 5', linewidth=line_widths[4], color='#800080', linestyle=line_styles[4])

# Configurações do terceiro subplot (frequência)
ax3.set_xlim(flims)
ax3.set_ylim((-80, 0))
ax3.grid(True)
ax3.legend(loc='lower right', fontsize=17)  # Legenda no canto inferior direito
ax3.set_xticks(ticks)
ax3.set_xticklabels([str(tick) for tick in ticks])
ax3.tick_params(axis='x', labelbottom=False)

# Quarto subplot (direita - Medições 6 a 10 na frequência)
ax4.semilogx(ppro6.freq_Hw, 20 * np.log10(np.abs(ppro6.Hww_mtx[0, :])), label='FRF med 6', linewidth=line_widths[0], color='#00FFFF', linestyle=line_styles[0])
ax4.semilogx(ppro7.freq_Hw, 20 * np.log10(np.abs(ppro7.Hww_mtx[0, :])), label='FRF med 7', linewidth=line_widths[1], color='#FFA500', linestyle=line_styles[1])
ax4.semilogx(ppro8.freq_Hw, 20 * np.log10(np.abs(ppro8.Hww_mtx[0, :])), label='FRF med 8', linewidth=line_widths[2], color='#FF00FF', linestyle=line_styles[2])
ax4.semilogx(ppro9.freq_Hw, 20 * np.log10(np.abs(ppro9.Hww_mtx[0, :])), label='FRF med 9', linewidth=line_widths[3], color='#008000', linestyle=line_styles[3])
ax4.semilogx(ppro10.freq_Hw, 20 * np.log10(np.abs(ppro10.Hww_mtx[0, :])), label='FRF med 10', linewidth=line_widths[4], color='#000000', linestyle=line_styles[4])

# Configurações do quarto subplot (frequência)
ax4.set_xlim(flims)
ax4.set_ylim((-80, 0))
ax4.set_xlabel('Frequency [Hz]', fontsize=28, fontweight='bold')
ax4.grid(True)
ax4.legend(loc='lower right', fontsize=17)  # Legenda no canto inferior direito
ax4.set_xticks(ticks)
ax4.set_xticklabels([str(tick) for tick in ticks])

# Remover rótulos Y dos subplots que não precisam
ax1.set_ylabel('')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax4.set_ylabel('')

# Adicionar um rótulo Y centralizado para todos os subplots
fig.text(0.001, 0.55, 'Amplitude [-]', va='center', rotation='vertical', fontsize=28, fontweight='bold')

# Ajustar espaçamento entre os subplots
plt.tight_layout()
fig.subplots_adjust(left=0.06)  # Aumenta a margem esquerda para o rótulo caber

# Exibir os gráficos
plt.show()


#%% Criar figura com 4 subplots, 2 colunas e 2 linhas
fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=(20, 14))

def format_ticks_ms(x,pos):
    return f'{x:.1f}'.replace('.', ',')

# Primeiro subplot (esquerda - Medições 1 a 5 no tempo)
ax1.plot(ppro1.time_ht * 1000, ppro1.ht_mtx[0, :] / np.amax(ppro1.ht_mtx[0, :]), label='RI med 1', linewidth=13.5, color='#FF0000',linestyle=':')
ax1.plot(ppro2.time_ht * 1000, ppro2.ht_mtx[0, :] / np.amax(ppro2.ht_mtx[0, :]), label='RI med 2', linewidth=5.5, color='#0000FF')
ax1.plot(ppro3.time_ht * 1000, ppro3.ht_mtx[0, :] / np.amax(ppro3.ht_mtx[0, :]), label='RI med 3', linewidth=8.8, color='#FFFF00',linestyle='-.')
ax1.plot(ppro4.time_ht * 1000, ppro4.ht_mtx[0, :] / np.amax(ppro4.ht_mtx[0, :]), label='RI med 4', linewidth=4.5, color='#00FF00',linestyle='--')
ax1.plot(ppro5.time_ht * 1000, ppro5.ht_mtx[0, :] / np.amax(ppro5.ht_mtx[0, :]), label='RI med 5', linewidth=1.7, color='#800080')

# Configurações do primeiro subplot (tempo)
ax1.set_xlim(0, tfim * 1000)
ax1.grid(True)
ax1.legend(loc='upper right',fontsize=17)  # Legenda no canto inferior direito
# ax1.set_xlabel(r'Tempo [s]', fontsize=22, fontweight='bold')
ax1.set_ylabel(r'Amplitude [-]', fontsize=22, fontweight='bold')
ax1.xaxis.set_major_formatter(FuncFormatter(format_ticks_x))
ax1.yaxis.set_major_formatter(FuncFormatter(format_ticks_y))

ax1.tick_params(axis='x', labelbottom=False)

# Segundo subplot (esquerda - Medições 6 a 10 no tempo)
ax2.plot(ppro6.time_ht * 1000, ppro6.ht_mtx[0, :] / np.amax(ppro6.ht_mtx[0, :]), label='RI med 6', linewidth=18.5, color='#00FFFF')
ax2.plot(ppro7.time_ht * 1000, ppro7.ht_mtx[0, :] / np.amax(ppro7.ht_mtx[0, :]), label='RI med 7', linewidth=13.5, color='#FFA500')
ax2.plot(ppro8.time_ht * 1000, ppro8.ht_mtx[0, :] / np.amax(ppro8.ht_mtx[0, :]), label='RI med 8', linewidth=8.8, color='#FF00FF')
ax2.plot(ppro9.time_ht * 1000, ppro9.ht_mtx[0, :] / np.amax(ppro9.ht_mtx[0, :]), label='RI med 9', linewidth=4.5, color='#008000')
ax2.plot(ppro10.time_ht * 1000, ppro10.ht_mtx[0, :] / np.amax(ppro10.ht_mtx[0, :]), label='RI med 10', linewidth=1.7, color='#000000')

# Configurações do segundo subplot (tempo)
ax2.set_xlim(0, tfim * 1000)
ax2.grid(True)
ax2.legend(loc='upper right',fontsize=17)  # Legenda no canto inferior direito
ax2.set_xlabel(r'Tempo [s]', fontsize=28, fontweight='bold')
ax2.set_ylabel(r'Amplitude [-]', fontsize=22, fontweight='bold')
ax2.xaxis.set_major_formatter(FuncFormatter(format_ticks_x))
ax2.yaxis.set_major_formatter(FuncFormatter(format_ticks_y))

ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_ticks_ms))

# Terceiro subplot (direita - Medições 1 a 5 na frequência)
ticks = [63,125,250,500,1000,2000,4000]
flims = (50, 4000)

ax3.semilogx(ppro1.freq_Hw, 20 * np.log10(np.abs(ppro1.Hww_mtx[0, :])), label='FRF med 1', linewidth=10.5, color='#FF0000')
ax3.semilogx(ppro2.freq_Hw, 20 * np.log10(np.abs(ppro2.Hww_mtx[0, :])), label='FRF med 2', linewidth=9.5, color='#0000FF')
ax3.semilogx(ppro3.freq_Hw, 20 * np.log10(np.abs(ppro3.Hww_mtx[0, :])), label='FRF med 3', linewidth=7.5, color='#FFFF00')
ax3.semilogx(ppro4.freq_Hw, 20 * np.log10(np.abs(ppro4.Hww_mtx[0, :])), label='FRF med 4', linewidth=4.5, color='#00FF00')
ax3.semilogx(ppro5.freq_Hw, 20 * np.log10(np.abs(ppro5.Hww_mtx[0, :])), label='FRF med 5', linewidth=1.7, color='#800080')

# Configurações do terceiro subplot (frequência)
ax3.set_xlim(flims)
ax3.set_ylim((-80, 0))
# ax3.set_xlabel('Frequência [Hz]', fontsize=22, fontweight='bold')
# ax3.set_ylabel('Amplitude [dB]', fontsize=22, fontweight='bold')
ax3.grid(True)
ax3.legend(loc='lower right',fontsize=17)  # Legenda no canto inferior direito
ax3.set_xticks(ticks)
ax3.set_xticklabels([str(tick) for tick in ticks])

ax3.tick_params(axis='x', labelbottom=False)

# Quarto subplot (direita - Medições 6 a 10 na frequência)
ax4.semilogx(ppro6.freq_Hw, 20 * np.log10(np.abs(ppro6.Hww_mtx[0, :])), label='FRF med 6', linewidth=10.5, color='#00FFFF')
ax4.semilogx(ppro7.freq_Hw, 20 * np.log10(np.abs(ppro7.Hww_mtx[0, :])), label='FRF med 7', linewidth=9.5, color='#FFA500')
ax4.semilogx(ppro8.freq_Hw, 20 * np.log10(np.abs(ppro8.Hww_mtx[0, :])), label='FRF med 8', linewidth=7.5, color='#FF00FF')
ax4.semilogx(ppro9.freq_Hw, 20 * np.log10(np.abs(ppro9.Hww_mtx[0, :])), label='FRF med 9', linewidth=4.5, color='#008000')
ax4.semilogx(ppro10.freq_Hw, 20 * np.log10(np.abs(ppro10.Hww_mtx[0, :])), label='FRF med 10', linewidth=1.7, color='#000000')

# Configurações do quarto subplot (frequência)
ax4.set_xlim(flims)
ax4.set_ylim((-80, 0))
ax4.set_xlabel('Frequência [Hz]', fontsize=28, fontweight='bold')
# ax4.set_ylabel('Amplitude [dB]', fontsize=22, fontweight='bold')
ax4.grid(True)
ax4.legend(loc='lower right',fontsize=17)  # Legenda no canto inferior direito
ax4.set_xticks(ticks)
ax4.set_xticklabels([str(tick) for tick in ticks])

ax1.set_ylabel('')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax4.set_ylabel('')

# Adicionar um rótulo Y centralizado para todos os subplots
fig.text(0.001, 0.55, 'Amplitude [-]', va='center', rotation='vertical', fontsize=28, fontweight='bold')

# Ajustar espaçamento entre os subplots
plt.tight_layout()
fig.subplots_adjust(left=0.06)  # Aumenta a margem esquerda para o rótulo caber

# Exibir os gráficos
plt.show()


#%% Plot FRFs

ticks = [63,125,250,500,1000,2000,4000]
flims = (50, 4000)

fig, ax = plt.subplots(figsize=(11, 7))

# Plotando todas as curvas no mesmo gráfico (semilogx)
ax.semilogx(ppro1.freq_Hw, 20 * np.log10(np.abs(ppro1.Hww_mtx[0, :])), label='FRF - PU', alpha = 0.7, linewidth=2.5, color=(0.267, 0.005, 0.329))
ax.semilogx(ppro2.freq_Hw, 20 * np.log10(np.abs(ppro2.Hww_mtx[0, :])), label='FRF - PU corrug', alpha = 0.7,linewidth=2.5, color=(0.127, 0.566, 0.550))
ax.semilogx(ppro3.freq_Hw, 20 * np.log10(np.abs(ppro3.Hww_mtx[1, :])), label='FRF - PU mounted',alpha = 0.7, linewidth=2.5, color=(0.993, 0.906, 0.144))
# ax.semilogx(ppro4.freq_Hw, 20 * np.log10(np.abs(ppro4.Hww_mtx[1, :])), label='FRF med 4', linewidth=10, color='green')
# ax.semilogx(ppro5.freq_Hw, 20 * np.log10(np.abs(ppro5.Hww_mtx[1, :])), label='FRF med 5', linewidth=8.5, color='orange')
# ax.semilogx(ppro6.freq_Hw, 20 * np.log10(np.abs(ppro6.Hww_mtx[1, :])), label='FRF med 6', linewidth=7, color='purple')
# ax.semilogx(ppro7.freq_Hw, 20 * np.log10(np.abs(ppro7.Hww_mtx[1, :])), label='FRF med 7', linewidth=6.4, color='brown')
# ax.semilogx(ppro8.freq_Hw, 20 * np.log10(np.abs(ppro8.Hww_mtx[1, :])), label='FRF med 8', linewidth=4, color='pink')
# ax.semilogx(ppro9.freq_Hw, 20 * np.log10(np.abs(ppro9.Hww_mtx[1, :])), label='FRF med 9', linewidth=3, color='black')
# ax.semilogx(ppro10.freq_Hw, 20 * np.log10(np.abs(ppro10.Hww_mtx[1, :])), label='FRF med 10', linewidth=1, color='red')

# Configurações dos eixos
ax.set_xlim(flims)
ax.set_ylim((-60, 0))
ax.set_xlabel('Frequency [Hz]',fontsize=22,fontweight='bold')
ax.set_ylabel('Amplitude [-]',fontsize=22,fontweight='bold')
ax.legend()

# Grades e ticks
ax.grid(True, alpha = 0.3)
ax.set_xticks(ticks)
ax.set_xticklabels([str(tick) for tick in ticks])
# plt.xticks(fontsize=26)
ax.tick_params(labelsize=26)
# ax.set_yticks(fontsize=26)
ax.legend(fontsize=20)

# Exibindo o gráfico
plt.show()

#%%

# Definindo os ticks e limites dos eixos
ticks = [63, 125, 250, 500, 1000, 2000, 4000]
flims = (50, 4000)

# Cores vivas e contrastantes
colors = ['#FF0000', '#0000FF', '#00FF00', '#FFA500', '#800080', '#00FFFF', '#A52A2A', '#FFC0CB', '#FFFF00', '#000000']
alphas = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Alta opacidade para cores mais vivas e nítidas

fig, ax = plt.subplots(figsize=(10.5, 9.5))

# Plotando todas as curvas no mesmo gráfico (semilogx) com cores contrastantes e opacidade alta
ax.semilogx(ppro1.freq_Hw, 20 * np.log10(np.abs(ppro1.Hww_mtx[0, :])), label='FRF med 1', linewidth=8, color=colors[0], alpha=alphas[0])
ax.semilogx(ppro2.freq_Hw, 20 * np.log10(np.abs(ppro2.Hww_mtx[0, :])), label='FRF med 2', linewidth=2, color=colors[1], alpha=alphas[1])
ax.semilogx(ppro3.freq_Hw, 20 * np.log10(np.abs(ppro3.Hww_mtx[1, :])), label='FRF med 3', linewidth=11, color=colors[2], alpha=alphas[2])
ax.semilogx(ppro4.freq_Hw, 20 * np.log10(np.abs(ppro4.Hww_mtx[1, :])), label='FRF med 4', linewidth=10, color=colors[3], alpha=alphas[3])
ax.semilogx(ppro5.freq_Hw, 20 * np.log10(np.abs(ppro5.Hww_mtx[1, :])), label='FRF med 5', linewidth=8.5, color=colors[4], alpha=alphas[4])
ax.semilogx(ppro6.freq_Hw, 20 * np.log10(np.abs(ppro6.Hww_mtx[1, :])), label='FRF med 6', linewidth=7, color=colors[5], alpha=alphas[5])
ax.semilogx(ppro7.freq_Hw, 20 * np.log10(np.abs(ppro7.Hww_mtx[1, :])), label='FRF med 7', linewidth=6.4, color=colors[6], alpha=alphas[6])
ax.semilogx(ppro8.freq_Hw, 20 * np.log10(np.abs(ppro8.Hww_mtx[1, :])), label='FRF med 8', linewidth=4, color=colors[7], alpha=alphas[7])
ax.semilogx(ppro9.freq_Hw, 20 * np.log10(np.abs(ppro9.Hww_mtx[1, :])), label='FRF med 9', linewidth=3, color=colors[8], alpha=alphas[8])
ax.semilogx(ppro10.freq_Hw, 20 * np.log10(np.abs(ppro10.Hww_mtx[1, :])), label='FRF med 10', linewidth=1, color=colors[9], alpha=alphas[9])

# Configurações dos eixos
ax.set_xlim(flims)
ax.set_ylim((-80, 0))
ax.set_xlabel('Frequência [Hz]', fontsize=22, fontweight='bold')
ax.set_ylabel('Amplitude [-]', fontsize=22, fontweight='bold')
ax.legend()

# Grades e ticks
ax.grid(True)
ax.set_xticks(ticks)
ax.set_xticklabels([str(tick) for tick in ticks])

# Exibindo o gráfico
plt.show()


#%% Plot IR + FRF

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,28), sharey=False, sharex=False)

# Rotacionando os ticks
for axis in ax.flat:
    axis.set_xticklabels(axis.get_xticks(), rotation=45)
    
# Plot das RI's e FRF's
ax[0,0].plot(ppro1.time_ht, ppro1.ht_mtx[0, :] / np.amax(ppro1.ht_mtx[0, :]), label='IR', linewidth=3, color='blue')
ax[0,0].grid(True)
ax[0,0].legend()
ax[0,0].set_xlim(0,tfim)
ax[0,0].set_ylim([-.5, 1.1])
ax[0,0].xaxis.set_major_formatter(FuncFormatter(format_ticks_x))
ax[0,0].yaxis.set_major_formatter(FuncFormatter(format_ticks_y))
ax[0,0].set_xticklabels([])  # Remove os ticks do eixo x

ax[0,1].semilogx(ppro1.freq_Hw, 20*np.log10(np.abs(ppro1.Hww_mtx[0, :])), label='FRF med 1', linewidth=4.5, color='blue')
ax[0,1].grid(True)
ax[0,1].legend()
ax[0,1].set_xlim(flims)
ax[0,1].set_ylim([-80, 0])
ax[0,1].set_xticklabels([])  # Remove os ticks do eixo x

# ax[1,0].plot(ppro5.time_ht, ppro5.ht_mtx[0, :] / np.amax(ppro5.ht_mtx[0, :]), label='RI med 5', linewidth=3.3, color='black')
# ax[1,0].set_xlabel('Tempo [s]', fontsize=22, fontweight='bold') 
# ax[1,0].grid(True)
# ax[1,0].legend()
# ax[1,0].set_xlim(0,tfim)
# ax[1,0].set_ylim([-.5, 1.1])
# ax[1,0].xaxis.set_major_formatter(FuncFormatter(format_ticks_x))
# ax[1,0].yaxis.set_major_formatter(FuncFormatter(format_ticks_y))
# ax[1,0].set_xticklabels([])  # Remove os ticks do eixo x

# ax[1,1].semilogx(ppro5.freq_Hw, 20*np.log10(np.abs(ppro5.Hww_mtx[0, :])), label='FRF med 5', linewidth=4.5, color='black')
# ax[1,1].set_xlabel('Frequência [Hz]', fontsize=22, fontweight='bold') 
# ax[1,1].set_xlim(flims)
# ax[1,1].set_ylim((-80, 0))
# ax[1,1].grid(True)
# ax[1,1].legend()
# ax[1,1].set_xticklabels([])  # Remove os ticks do eixo x

# ax[2,0].plot(ppro10.time_ht, ppro10.ht_mtx[0, :] / np.amax(ppro10.ht_mtx[0, :]), label='RI med 10', linewidth=3.3, color='red')
# ax[2,0].set_xlabel('Tempo [s]', fontsize=22, fontweight='bold') 
# ax[2,0].grid(True)
# ax[2,0].legend()
# ax[2,0].set_xlim(0,tfim)
# ax[2,0].set_ylim([-.5, 1.1])
# ax[2,0].xaxis.set_major_formatter(FuncFormatter(format_ticks_x))
# ax[2,0].yaxis.set_major_formatter(FuncFormatter(format_ticks_y))

# ax[2,1].semilogx(ppro10.freq_Hw, 20*np.log10(np.abs(ppro10.Hww_mtx[0, :])), label='FRF med 10', linewidth=4.5, color='red')
# ax[2,1].set_xlabel('Frequência [Hz]', fontsize=22, fontweight='bold') 
# ax[2,1].set_xlim(flims)
# ax[2,1].set_ylim((-80, 0))
# ax[2,1].grid(True)
# ax[2,1].legend()

# Ajustando o espaçamento para evitar sobreposição
fig.text(0.03, 0.5, 'Amplitude [-]', va='center', ha='center', rotation='vertical', fontsize=22, fontweight='bold')

# Ajuste de espaçamento entre subplots e margens
plt.subplots_adjust(hspace=0.4, wspace=0.3, left=0.08, right=0.95, top=0.95, bottom=0.08)

# Adicionando os ticks corretos na Frequência
ax[0,1].set_xticks(ticks)
ax[0,1].set_xticklabels([str(tick) for tick in ticks])
# ax[1,1].set_xticks(ticks)
# ax[1,1].set_xticklabels([str(tick) for tick in ticks])
# ax[2,1].set_xticks(ticks)
# ax[2,1].set_xticklabels([str(tick) for tick in ticks])

# # Rotacionando os ticks
# for axis in ax.flat:
#     axis.set_xticklabels(axis.get_xticks(), rotation=45)
    
# Salvando a figura para evitar corte de labels e legendas
# plt.savefig('RI_FRF.pdf', bbox_inches='tight')

# plt.show()

#%% Cálculo de absorção

""" Import da classe AirProperties, atribuindo valores de 
    velocidade do som c0 e densidade rho0 a uma variável = air
    
    Seguidamente a classe de controle eh utilizada, criando objetos
    de controle com velocidade(c0), densidade(rho0) e vetor de 
    frequencias(freq_vec) definido pelo vetor de frequências do 
    objeto de pos processamento(ppro).
    
"""
air = AirProperties(c0 = 343.0, rho0 = 1.21,)
controls1 = AlgControls(c0 = air.c0, freq_vec = ppro1.freq_Hw) 
controls2 = AlgControls(c0 = air.c0, freq_vec = ppro2.freq_Hw) 
controls3 = AlgControls(c0 = air.c0, freq_vec = ppro3.freq_Hw) 
controls4 = AlgControls(c0 = air.c0, freq_vec = ppro4.freq_Hw) 
controls5 = AlgControls(c0=air.c0, freq_vec=ppro5.freq_Hw)
controls6 = AlgControls(c0=air.c0, freq_vec=ppro6.freq_Hw)
controls7 = AlgControls(c0=air.c0, freq_vec=ppro7.freq_Hw)
controls8 = AlgControls(c0=air.c0, freq_vec=ppro8.freq_Hw)
controls9 = AlgControls(c0=air.c0, freq_vec=ppro9.freq_Hw)
controls10 = AlgControls(c0=air.c0, freq_vec=ppro10.freq_Hw)

subs = np.zeros(ppro2.Hww_mtx.shape, dtype = complex)
subs[0,:] = ppro2.Hww_mtx[0,:]
subs[1,:] = ppro2.Hww_mtx[1,:]

""" A classe ImpedanceDeductionQterm eh acionada. Se trata de uma
"""

h_pp1 = ImpedanceDeductionQterm(p_mtx=ppro1.Hww_mtx, controls=controls1, 
                                receivers=ppro1.meas_obj.receivers, 
                                source=ppro1.meas_obj.source)
h_pp1.pw_pp()
h_pp1.pwa_pp()

# h_pp2 = ImpedanceDeductionQterm(p_mtx=subs, controls=controls2, 
#                                 receivers=ppro2.meas_obj.receivers, 
#                                 source=ppro2.meas_obj.source)
# h_pp2.pw_pp()
# h_pp2.pwa_pp()

# h_pp3 = ImpedanceDeductionQterm(p_mtx=subs, controls=controls3, 
#                                 receivers=ppro3.meas_obj.receivers, 
#                                 source=ppro3.meas_obj.source)
# h_pp3.pw_pp()
# h_pp3.pwa_pp()

# h_pp4 = ImpedanceDeductionQterm(p_mtx=subs, controls=controls4, 
#                                 receivers=ppro4.meas_obj.receivers, 
#                                 source=ppro4.meas_obj.source)
# h_pp4.pw_pp()
# h_pp4.pwa_pp()


# h_pp5 = ImpedanceDeductionQterm(p_mtx=subs, controls=controls4, 
#                                 receivers=ppro5.meas_obj.receivers, 
#                                 source=ppro5.meas_obj.source)
# h_pp5.pw_pp()
# h_pp5.pwa_pp()

# h_pp6 = ImpedanceDeductionQterm(p_mtx=subs, controls=controls6, 
#                                 receivers=ppro6.meas_obj.receivers, 
#                                 source=ppro6.meas_obj.source)
# h_pp6.pw_pp()
# h_pp6.pwa_pp()

# h_pp7 = ImpedanceDeductionQterm(p_mtx=subs, controls=controls7, 
#                                 receivers=ppro7.meas_obj.receivers, 
#                                 source=ppro7.meas_obj.source)
# h_pp7.pw_pp()
# h_pp7.pwa_pp()

# h_pp8 = ImpedanceDeductionQterm(p_mtx=subs, controls=controls8, 
#                                 receivers=ppro8.meas_obj.receivers, 
#                                 source=ppro8.meas_obj.source)
# h_pp8.pw_pp()
# h_pp8.pwa_pp()

# h_pp9 = ImpedanceDeductionQterm(p_mtx=subs, controls=controls9, 
#                                 receivers=ppro9.meas_obj.receivers, 
#                                 source=ppro9.meas_obj.source)
# h_pp9.pw_pp()
# h_pp9.pwa_pp()

# h_pp10 = ImpedanceDeductionQterm(p_mtx=subs, controls=controls10, 
#                                   receivers=ppro10.meas_obj.receivers, 
#                                   source=ppro10.meas_obj.source)
# h_pp10.pw_pp()
# h_pp10.pwa_pp()

h_pp2 = ImpedanceDeductionQterm(p_mtx=ppro2.Hww_mtx, controls=controls2, 
                                receivers=ppro2.meas_obj.receivers, 
                                source=ppro2.meas_obj.source)
h_pp2.pw_pp()
h_pp2.pwa_pp()

h_pp3 = ImpedanceDeductionQterm(p_mtx=ppro3.Hww_mtx, controls=controls3, 
                                receivers=ppro3.meas_obj.receivers, 
                                source=ppro3.meas_obj.source)
h_pp3.pw_pp()
h_pp3.pwa_pp()

h_pp4 = ImpedanceDeductionQterm(p_mtx=ppro4.Hww_mtx, controls=controls4, 
                                receivers=ppro4.meas_obj.receivers, 
                                source=ppro4.meas_obj.source)
h_pp4.pw_pp()
h_pp4.pwa_pp()


h_pp5 = ImpedanceDeductionQterm(p_mtx=ppro5.Hww_mtx, controls=controls4, 
                                receivers=ppro5.meas_obj.receivers, 
                                source=ppro5.meas_obj.source)
h_pp5.pw_pp()
h_pp5.pwa_pp()

h_pp6 = ImpedanceDeductionQterm(p_mtx=ppro6.Hww_mtx, controls=controls6, 
                                receivers=ppro6.meas_obj.receivers, 
                                source=ppro6.meas_obj.source)
h_pp6.pw_pp()
h_pp6.pwa_pp()

h_pp7 = ImpedanceDeductionQterm(p_mtx=ppro7.Hww_mtx, controls=controls7, 
                                receivers=ppro7.meas_obj.receivers, 
                                source=ppro7.meas_obj.source)
h_pp7.pw_pp()
h_pp7.pwa_pp()

h_pp8 = ImpedanceDeductionQterm(p_mtx=ppro8.Hww_mtx, controls=controls8, 
                                receivers=ppro8.meas_obj.receivers, 
                                source=ppro8.meas_obj.source)
h_pp8.pw_pp()
h_pp8.pwa_pp()

h_pp9 = ImpedanceDeductionQterm(p_mtx=ppro9.Hww_mtx, controls=controls9, 
                                receivers=ppro9.meas_obj.receivers, 
                                source=ppro9.meas_obj.source)
h_pp9.pw_pp()
h_pp9.pwa_pp()

h_pp10 = ImpedanceDeductionQterm(p_mtx=ppro10.Hww_mtx, controls=controls10, 
                                  receivers=ppro10.meas_obj.receivers, 
                                  source=ppro10.meas_obj.source)
h_pp10.pw_pp()
h_pp10.pwa_pp()

#%% Plot de absorção - 10 meds no mesmo gráfico
plt.figure(figsize=(16,9))
plt.semilogx(h_pp1.controls.freq, h_pp1.alpha_pwa_pp, label = 'Med. 01',color = 'blue', linewidth = 2, alpha = 0.7)
plt.semilogx(h_pp2.controls.freq, h_pp2.alpha_pwa_pp, label = 'Med. 02',color = 'green', linewidth = 2, alpha = 0.7)
plt.semilogx(h_pp3.controls.freq, h_pp1.alpha_pwa_pp, label = 'Med. 03*',color='red', linewidth = 2, alpha = 0.7)
plt.semilogx(h_pp4.controls.freq, h_pp4.alpha_pwa_pp, label = 'Med. 04',color = 'orange', linewidth = 2, alpha = 0.7)
plt.semilogx(h_pp5.controls.freq, h_pp5.alpha_pwa_pp, label = 'Med. 05',color = 'purple', linewidth = 2, alpha = 0.7)
plt.semilogx(h_pp6.controls.freq, h_pp6.alpha_pwa_pp, label = 'Med. 06',color = 'yellow', linewidth = 2, alpha = 0.7)
plt.semilogx(h_pp7.controls.freq, h_pp7.alpha_pwa_pp, label = 'Med. 07',color = 'cyan', linewidth = 2, alpha = 0.7)
plt.semilogx(h_pp8.controls.freq, h_pp8.alpha_pwa_pp, label = 'Med. 08',color = 'magenta', linewidth = 2, alpha = 0.7)
plt.semilogx(h_pp9.controls.freq, h_pp9.alpha_pwa_pp, label = 'Med. 09',color = 'lime', linewidth = 2, alpha = 0.7)
plt.semilogx(h_pp10.controls.freq, h_pp10.alpha_pwa_pp, label = 'Med. 10',color = 'black', linewidth = 2, alpha = 0.7)
plt.semilogx(h_pp10.controls.freq, np.zeros(len(h_pp10.alpha_pwa_pp)), color = 'grey', linewidth = 3, alpha = 1, linestyle = '--')
plt.legend(loc='lower right', fontsize = 18)
# Ajusta a espessura das linhas na legenda
handles, labels = plt.gca().get_legend_handles_labels()
for handle in handles:
    handle.set_linewidth(3)  # Define a espessura desejada para as linhas na legenda
plt.ylim((-0.2, 1.0))
plt.xlim((100, 4000))
plt.grid()
plt.xticks(ticks = [125,250,500,1000,2000,4000], labels = ['125','250','500','1000','2000','4000'], rotation = 0)
plt.xticks(fontsize=22)
plt.title('Coeficiente de absorção inferido pelo modelo PWA')
#plt.xlabel(r'Frequency [Hz]')
#plt.ylabel(r'$\alpha$ [-]')
plt.xlabel(r'Frequência [Hz]', fontsize=22, fontweight='bold')
plt.ylabel(r'$\alpha$ [-]', fontsize=22, fontweight='bold')

# Aplicar o formatador personalizado aos eixos x e y
# plt.gca().xaxis.set_major_formatter(FuncFormatter(format_ticks_x))
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_ticks_y))


#plt.tight_layout();
#plt.savefig("Coef_abs_29072024.pdf", format="pdf")
   

############################################################################
############################################################################
############################################################################

#%%
import matplotlib.colors as mcolors

n_medicoes = 10

# Lista de medições e frequências
medicoes = [h_pp1, h_pp2, h_pp3, h_pp4, h_pp5, h_pp6, h_pp7, h_pp8, h_pp9, h_pp10]
frequencias = h_pp1.controls.freq  # Presumindo que todas as medições compartilham o mesmo vetor de frequências

# Criando uma matriz com os valores de alpha_pwa_pp para cada medição
alpha_matrix = np.array([med.alpha_pwa_pp for med in medicoes])

# Definir o número de medições para o eixo Y
n_medicoes = alpha_matrix.shape[0]

# Criando o plot
fig, ax = plt.subplots(figsize=(14, 9))

# Mapa de cores (heatmap) das medições
c = ax.pcolormesh(frequencias, np.arange(1, n_medicoes + 1), alpha_matrix, cmap='inferno', shading='auto', norm=mcolors.Normalize(vmin=-0.4, vmax=1.0))

# Adicionando linhas tracejadas para dividir as medições
for i in range(1, n_medicoes):  # Adiciona uma linha para cada separação entre medições
    ax.hlines(i + 0.5, xmin=frequencias[0], xmax=frequencias[-1], color='black', linestyle='--', linewidth=3)

# Adicionando uma barra de cores para mostrar a escala de α (coeficiente de absorção)
cbar = fig.colorbar(c, ax=ax)
cbar.set_label(r'Coeficiente de absorção $\alpha$ [-]', fontsize=22, fontweight='bold')
cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_ticks_y))  # Aplicar o formatador personalizado

# Configurações dos eixos
ax.set_xscale('log')  # Escala logarítmica no eixo X (frequências)
ax.set_xlim(100, 4000)  # Definir os limites de frequência
ax.set_ylim(0.5, n_medicoes + 0.5)  # Ajustando o limite do eixo Y

# Personalizando os rótulos dos eixos
ax.set_xlabel(r'Frequência [Hz]', fontsize=22, fontweight='bold')
ax.set_ylabel('Medições', fontsize=22, fontweight='bold')

# Adicionando os rótulos das medições no eixo Y
ax.set_yticks(np.arange(1, n_medicoes + 1))
ax.set_yticklabels([f'Med {i + 1}' for i in range(n_medicoes)])

# Personalizando os ticks de frequência
ax.set_xticks([125, 250, 500, 1000, 2000, 4000])
ax.set_xticklabels(['125', '250', '500', '1000', '2000', '4000'])

# Adicionando grades ao gráfico
ax.grid(False)

output_file = "grafico_otimizado.jpg"  # Ou use .png para gráficos sem perdas

# Usando JPEG com compressão (qualidade de 85)
output_file = "grafico_otimizado.png"
fig.savefig(output_file, format='png', optimize=True, dpi=100)
# Mostrando o gráfico
plt.show()



#%%#%% Plot FRF's

# ticks = [63,125,250,500,1000,2000,4000]
# flims = (50, 4000)
# fig, ax = plt.subplots(nrows=2,ncols=2 figsize = (8,6), sharex = False)

# fig, ax = plt.plot(figsize = (8,6))
# ax[0].semilogx(ppro1.freq_Hw, 20*np.log10(np.abs(ppro1.Hww_mtx[0, :])), label = 'med 1')
# ax[0].semilogx(ppro2.freq_Hw, 20*np.log10(np.abs(ppro2.Hww_mtx[0, :])), label = 'med 2')
# ax[0].set_xlim(flims)
# ax[0].set_ylim((-80, 10))
# ax[0].legend()
# ax[1].semilogx(ppro3.freq_Hw, 20*np.log10(np.abs(ppro3.Hww_mtx[1, :])), label = 'med 3')
# ax[1].semilogx(ppro8.freq_Hw, 20*np.log10(np.abs(ppro8.Hww_mtx[1, :])), label = 'med 8')
# ax[1].set_xlim(flims)
# ax[1].set_ylim((-80, 10))
# ax[1].legend()
# ax[0].grid(True)
# ax[1].grid(True)

# ax[0].set_xticks(ticks)
# ax[0].set_xticklabels([str(tick) for tick in ticks])
# ax[1].set_xticks(ticks)
# ax[1].set_xticklabels([str(tick) for tick in ticks])
#%% Absorção

# Cria a figura e os dois subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14), sharex=True)

# Primeiro subplot (medições 1 a 5)
ax1.semilogx(h_pp1.controls.freq, h_pp1.alpha_pwa_pp, label='med 1 - PWA', color='#FF0000', linewidth=13.5, alpha=0.7)
ax1.semilogx(h_pp2.controls.freq, h_pp2.alpha_pwa_pp, label='med 2 - PWA', color='#0000FF', linewidth=13.5, alpha=0.5)
ax1.semilogx(h_pp3.controls.freq, h_pp3.alpha_pwa_pp, label='med 3 - PWA', color='#FFFF00', linewidth=8.8, alpha=0.5)
ax1.semilogx(h_pp4.controls.freq, h_pp4.alpha_pwa_pp, label='med 4 - PWA', color='#00FF00', linewidth=4.5, alpha=0.7)
ax1.semilogx(h_pp5.controls.freq, h_pp5.alpha_pwa_pp, label='med 5 - PWA', color='#800080', linewidth=1.7, alpha=0.7)
ax1.legend(loc='lower right', fontsize=20)
ax1.set_ylim((-0.4, 1.0))
ax1.grid()


# Segundo subplot (medições 6 a 10)
ax2.semilogx(h_pp6.controls.freq, h_pp6.alpha_pwa_pp, label='med 6 - PWA', color='#00FFFF', linewidth=18.5, alpha=0.7)
ax2.semilogx(h_pp7.controls.freq, h_pp7.alpha_pwa_pp, label='med 7 - PWA', color='#FFA500', linewidth=13.5, alpha=0.7)
ax2.semilogx(h_pp8.controls.freq, h_pp8.alpha_pwa_pp, label='med 8 - PWA', color='#FF00FF', linewidth=8.8, alpha=0.7)
ax2.semilogx(h_pp9.controls.freq, h_pp9.alpha_pwa_pp, label='med 9 - PWA', color='#008000', linewidth=4.5, alpha=0.7)
ax2.semilogx(h_pp10.controls.freq, h_pp10.alpha_pwa_pp, label='med 10 - PWA', color='#000000', linewidth=1.7, alpha=0.7)
ax2.legend(loc='lower right', fontsize=20)
ax2.set_ylim((-0.4, 1.0))
ax2.grid()

# Configurações compartilhadas para ambos os subplots
ax2.set_xlim((100, 4000))
ax2.set_xticks([125, 250, 500, 1000, 2000, 4000])
ax2.set_xticklabels(['125', '250', '500', '1000', '2000', '4000'], fontsize=22)
ax2.set_xlabel(r'Frequência [Hz]', fontsize=22, fontweight='bold')
ax1.set_ylabel(r'$\alpha$ [-]', fontsize=22, fontweight='bold')
ax2.set_ylabel(r'$\alpha$ [-]', fontsize=22, fontweight='bold')

# Aplicar o formatador personalizado ao eixo y
ax1.yaxis.set_major_formatter(FuncFormatter(format_ticks_y))
ax2.yaxis.set_major_formatter(FuncFormatter(format_ticks_y))

# Mostrar o layout
plt.tight_layout()
# plt.savefig("Coef_abs_29072024.pdf", format="pdf")
#################################################################



#%% OS DOIS JUNTOS

# Função para formatar valores com vírgula
def format_ticks(x, pos):
    return f'{x:.2f}'.replace('.', ',')

# Configurando os dados de exemplo
n_medicoes = 10
medicoes = [h_pp1, h_pp2, h_pp3, h_pp4, h_pp5, h_pp6, h_pp7, h_pp8, h_pp9, h_pp10]
frequencias = h_pp1.controls.freq  # Presumindo que todas as medições compartilham o mesmo vetor de frequências
alpha_matrix = np.array([med.alpha_pwa_pp for med in medicoes])

# Criando a estrutura de subplots
fig, axs = plt.subplots(2, 1, figsize=(18, 20), sharex=False)

# Primeiro gráfico: Curvas com transparência
axs[0].semilogx(h_pp1.controls.freq, h_pp1.alpha_pwa_pp, label='med 1 - PWA', color='blue', linewidth=4, alpha=1)
axs[0].semilogx(h_pp2.controls.freq, h_pp2.alpha_pwa_pp, label='med 2 - PWA', color='green', linewidth=4, alpha=0.9)
axs[0].semilogx(h_pp3.controls.freq, h_pp3.alpha_pwa_pp, label='med 3 - PWA', color='red', linewidth=4, alpha=0.8)
axs[0].semilogx(h_pp4.controls.freq, h_pp4.alpha_pwa_pp, label='med 4 - PWA', color='orange', linewidth=4, alpha=0.7)
axs[0].semilogx(h_pp5.controls.freq, h_pp5.alpha_pwa_pp, label='med 5 - PWA', color='purple', linewidth=4, alpha=0.6)
axs[0].semilogx(h_pp6.controls.freq, h_pp6.alpha_pwa_pp, label='med 6 - PWA', color='yellow', linewidth=4, alpha=0.5)
axs[0].semilogx(h_pp7.controls.freq, h_pp7.alpha_pwa_pp, label='med 7 - PWA', color='cyan', linewidth=4, alpha=0.4)
axs[0].semilogx(h_pp8.controls.freq, h_pp8.alpha_pwa_pp, label='med 8 - PWA', color='magenta', linewidth=4, alpha=0.3)
axs[0].semilogx(h_pp9.controls.freq, h_pp9.alpha_pwa_pp, label='med 9 - PWA', color='lime', linewidth=4, alpha=0.2)
axs[0].semilogx(h_pp10.controls.freq, h_pp10.alpha_pwa_pp, label='med 10 - PWA', color='black', linewidth=4, alpha=0.2)
axs[0].legend(loc='lower right', fontsize=14)
axs[0].set_ylim((-0.4, 1.0))
axs[0].set_xlim((100, 4000))
axs[0].grid()
axs[0].set_xlabel(r'Frequência [Hz]', fontsize=22, fontweight='bold')
axs[0].set_ylabel(r'$\alpha$ [-]', fontsize=22, fontweight='bold')
axs[0].set_xticks([125, 250, 500, 1000, 2000, 4000])
axs[0].set_xticklabels(['125', '250', '500', '1000', '2000', '4000'])
axs[0].tick_params(axis='x', labelsize=18)
axs[0].tick_params(axis='y', labelsize=18)

# Segundo gráfico: Heatmap com linhas tracejadas
cbar = None
c = axs[1].pcolormesh(frequencias, np.arange(1, n_medicoes + 1), alpha_matrix, cmap='inferno', shading='auto', norm=mcolors.Normalize(vmin=-0.4, vmax=1.0))

# Adicionando linhas tracejadas para dividir as medições
for i in range(1, n_medicoes):
    axs[1].hlines(i + 0.5, xmin=frequencias[0], xmax=frequencias[-1], color='black', linestyle='--', linewidth=3)

# Adicionando uma barra de cores para mostrar a escala de α (coeficiente de absorção)
cbar = fig.colorbar(c, ax=axs[1])
cbar.set_label(r'Coeficiente de absorção $\alpha$ [-]', fontsize=22, fontweight='bold')
cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_ticks))  # Aplicar o formatador personalizado

# Configurações dos eixos
axs[1].set_xscale('log')  # Escala logarítmica no eixo X (frequências)
axs[1].set_xlim(100, 4000)  # Definir os limites de frequência
axs[1].set_ylim(0.5, n_medicoes + 0.5)  # Ajustando o limite do eixo Y
axs[1].set_xlabel(r'Frequência [Hz]', fontsize=22, fontweight='bold')
axs[1].set_ylabel('Medições', fontsize=22, fontweight='bold')
axs[1].set_yticks(np.arange(1, n_medicoes + 1))
axs[1].set_yticklabels([f'Med {i + 1}' for i in range(n_medicoes)])
axs[1].set_xticks([125, 250, 500, 1000, 2000, 4000])
axs[1].set_xticklabels(['125', '250', '500', '1000', '2000', '4000'])
axs[1].tick_params(axis='x', labelsize=18)
axs[1].tick_params(axis='y', labelsize=18)
axs[1].grid(False)

# Ajustando layout
plt.tight_layout()

# Mostrando o gráfico
plt.show()


