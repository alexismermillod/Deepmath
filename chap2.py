# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 22:16:39 2021
Le module numpy aide à effectuer des calculs numériques efficacement
@author: Alexis
"""

# on trace graphe de la fonction x -> cos (x) sur [0, 10]

# Import math Library
import math

# on le renomme np de façon raccourcie
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0, 2 * math.pi, 100)
Y = np.cos(X)

plt.plot(X, Y)
plt.show()

# Definition d'un vecteur
# Les principaux objets de numpy sont des tableaux. Les tableaux à une dimension sont des vecteurs

X = np.array([1,2,3,4])
print(X)
# Le type de X est numpy.ndarray
type(X)

# Definition d'une suite d'éléments avec arange()
# possibilité d'utiliser un pas non entier contrairement a range()

np.arange(1,8,0.5)

# Definition d'une division d'intervalle avec linspace()
# obtenir une subdivision régulière [a,b] de n valeurs (n-1 sous-intervalles)

np.linspace(0, 1, 10)

# Opérations élémentaires
# définition d'un vecteur
X = np.array([1,2,3,4])
# multiplication par un scalaire
Y = 2 * X
# addition d'une constante 
Y = X + 1
# carré 
Y = X ** 2
# somme 
np.sum(X)
# min, max
min = np.min(X)
max = np.max(X)

# Utilisation comme liste
X = np.linspace(1, 2, num=10)
X[0]
len(X)

# parcours de tous les éméments (deux méthodes)
for x in X:
    print(x)
    
for i in range(len(X)):
    print(i,X[i])
        
# Application d'une fonction 
# avec numpy on peut appliquer fonction sur chaque coordonnée d'un vecteur
X = np.array([0,1,2,3])
np.sqrt(X)

# Matplotlib
def f(x):
    return np.exp(-x**2)

a,b = -3,3
X = np.linspace(a, b, num=100)
Y = f(X)

plt.plot(X, Y)
plt.show()

# Tracé des fonctions points par points
X = np.array([0,1,2,3])
Y = np.array([0,2,4,1])
plt.plot(X, Y)
plt.plot(X, Y,linewidth=1,color='red')

# Axes 

def f(x):
    return np.exp(-x) * np.cos(2*np.pi*x)

a,b = 0,5
X = np.linspace(a, b, num=100)
Y = f(X)

plt.title('Amorti')
plt.axis('equal') # repère orthonormé
plt.grid() # grille
plt.xlim(a,b) # bornes de l'axe des x
plt.plot(X,Y)
plt.savefig('amorti.png') # sauvegarder l'image
plt.show()









 
