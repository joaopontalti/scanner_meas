# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:23:22 2025

@author: joaop

"""
x = "olá Joãozinho"
alpha = (41 + 73 - 25) ** 2 


class bota(object):
    saida = alpha
    
bota.saida
#%%
'''
    Aqui estou criando uma variável x em linha e determinando que o x da minha classe
'''
#%%
def mymethod(self):
    return self.x

x = "ta e agora"

class Brilha(object):
    x = x
    mymethod = mymethod
    
Brilha().mymethod()