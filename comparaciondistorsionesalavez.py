# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 02:56:03 2025

@author: jandr
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy import zeros, cos,sin
from matplotlib.patches import FancyArrowPatch

ptildeesp=np.array([[ -3.6473917,   -4.77908066],
 [ -3.59808473,  -4.03613015],
 [ -4.53882806,  -3.40259502],
 [ -6.00538958,  -3.93407789],
 [ -6.63353316,  -5.77010125],
 [ -5.39594174,  -7.47687463],
 [ -3.00102552,  -7.41954128],
 [ -1.59876946,  -5.05271252],
 [ -3.01671918,  -1.82929595],
 [ -6.89969863,  -0.56486047],
 [-10.66741188,  -2.77637961],
 [-11.44125922,  -7.17269137],
 [ -8.09434157, -10.37757889],
 [ -2.98977958,  -9.44046928],
 [ -0.28249902,  -4.36627936]])

n = 15  # número de agentes
a = 0.5  # radio inicial
b = 0.5  # separación entre vueltas
theta = np.linspace(0, 4 * np.pi, n)  # ángulos equiespaciados

r = a + b * theta
x = r * np.cos(theta)
y = r * np.sin(theta)

# Vector columna p_ref (ordenado como [x1, y1, x2, y2, ..., xn, yn])
p_refesp = np.empty((2*n, 1))
p_refesp[0::2, 0] = x
p_refesp[1::2, 0] = y



ptildecua=np.array([[ 2.29450795,  1.98528797],
 [-0.06264526,  1.89592418],
 [ 0.01670613, -0.26380972],
 [ 1.94091747, -0.4804344 ]])

prefcua=np.array([ 
    [1], [1],  # Agente 1
    [-1], [1],  # Agente 2
    [-1], [-1],  # Agente 3
    [1], [-1]  # Agente 4
])



# Ya definidos:
# - ptildeesp (15, 2)
# - p_refesp (30, 1) → hay que convertirlo
# - ptildecua (4, 2)
# - prefcua (8, 1) → hay que convertirlo

# Convertir p_refesp y prefcua a formato (n, 2)
p_refesp_2D = p_refesp.reshape(-1, 2)
prefcua_2D = prefcua.reshape(-1, 2)

# Crear figura y subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Primer subplot: formación cuadrada
n_cua = prefcua_2D.shape[0]
axs[0].plot(prefcua_2D[:, 0], prefcua_2D[:, 1], 'bo-', label='Formación de referencia cuadrada')
axs[0].plot(ptildecua[:, 0], ptildecua[:, 1], 'ro--', label='Formación distorsionada alcanzada')
for i in range(n_cua):
    axs[0].text(prefcua_2D[i, 0] + 0.05, prefcua_2D[i, 1] + 0.05, f'{i+1}', color='blue')
    axs[0].text(ptildecua[i, 0] + 0.05, ptildecua[i, 1] - 0.1, f'{i+1}', color='red')
axs[0].set_title('Formación de referencia cuadrada vs. distorsionada')
axs[0].set_xlabel('Posición X(-)')
axs[0].set_ylabel('Posición Y(-)')
axs[0].axis('equal')
axs[0].grid(True)
axs[0].legend()

# Segundo subplot: formación espiral
n_esp = p_refesp_2D.shape[0]
axs[1].plot(p_refesp_2D[:, 0], p_refesp_2D[:, 1], 'bo-', label='Formación de referncia, espiral')
axs[1].plot(ptildeesp[:, 0], ptildeesp[:, 1], 'ro--', label='Formación distorsionada alcanzada')
for i in range(n_esp):
    axs[1].text(p_refesp_2D[i, 0] + 0.05, p_refesp_2D[i, 1] + 0.05, f'{i+1}', color='blue')
    axs[1].text(ptildeesp[i, 0] + 0.05, ptildeesp[i, 1] - 0.1, f'{i+1}', color='red')
axs[1].set_title('Formación de referencia en espiral vs. distorsionada')
axs[1].set_xlabel('Posición X(-)')
axs[1].set_ylabel('Posición Y(-)')
axs[1].axis('equal')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()









#para pintar comparacion de normas en el caso con las imperfecciones en la misma gráfica
n = 15  # número de agentes
a = 0.5  # radio inicial
b = 0.5  # separación entre vueltas
theta = np.linspace(0, 4 * np.pi, n)  # ángulos equiespaciados

r = a + b * theta
x = r * np.cos(theta)
y = r * np.sin(theta)

# Vector columna p_ref (ordenado como [x1, y1, x2, y2, ..., xn, yn])
p_refesp = np.empty((2*n, 1))
p_refesp[0::2, 0] = x
p_refesp[1::2, 0] = y

ptilde_ref_1=np.array([[ 2.29450795],
 [ 1.98528797],
 [-0.06264526],
 [ 1.89592418],
 [ 0.01670613],
 [-0.26380972],
 [ 1.94091747],
 [-0.4804344 ]])

vtilde_ref_1=np.array([[-0.59409723],
 [ 0.05735551]])

data = np.load('datos_caso1.npz')

# Acceder a las variables
traj_1= data['traj']
pdots_1= data['pdots']
ptilde_ref_1= data['ptilde_ref']
Bbarra_1= data['Bbarra']
vtilde_ref_1= data['vtilde_ref']

data2= np.load('datos_caso2.npz')

# Acceder a las variables
traj_2= data2['traj']
pdots_2= data2['pdots']
ptilde_ref_2= data2['ptilde_ref']
Bbarra_2= data2['Bbarra']
vtilde_ref_2= data2['vtilde_ref']








dt=0.1

# ========= CASO 1 =========
n1 = ptilde_ref_1.shape[0] // 2
ztilde_ref_1 = Bbarra_1.T @ ptilde_ref_1

errores_1 = []
for p_t in traj_1:
    z_t = Bbarra_1.T @ p_t.reshape((-1, 1))
    error = np.linalg.norm(z_t - ztilde_ref_1)
    errores_1.append(error)

vtilde_ref_global_1 = np.kron(np.ones((n1, 1)), vtilde_ref_1)
errores_velocidad_1 = []
for pdot_t in pdots_1:
    error_v = np.linalg.norm(pdot_t - vtilde_ref_global_1)
    errores_velocidad_1.append(error_v)
errores_1 = errores_1[1:]
errores_velocidad_1 = errores_velocidad_1

tiempos_1 = np.arange(len(errores_1)) * dt
print(len(tiempos_1),len(errores_1),len(errores_velocidad_1))

# ========= CASO 2 =========
dt=0.01

n2 = ptilde_ref_2.shape[0] // 2
ztilde_ref_2 = Bbarra_2.T @ ptilde_ref_2

errores_2 = []
for p_t in traj_2:
    z_t = Bbarra_2.T @ p_t.reshape((-1, 1))
    error = np.linalg.norm(z_t - ztilde_ref_2)
    errores_2.append(error)

vtilde_ref_global_2 = np.kron(np.ones((n2, 1)), vtilde_ref_2)
errores_velocidad_2 = []
for pdot_t in pdots_2:
    error_v = np.linalg.norm(pdot_t - vtilde_ref_global_2)
    errores_velocidad_2.append(error_v)

errores_2 = errores_2[1:]
errores_velocidad_2 = errores_velocidad_2
tiempos_2 = np.arange(len(errores_2)) * dt

# ========= GRAFICADO COMPARATIVO =========
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(tiempos_1, errores_1, label=r'$\|z(t) - \tilde{z}^*\|$(-)')
axs[0].plot(tiempos_1, errores_velocidad_1, label=r'$\|\dot{p}(t) - (\mathbf{1} \otimes \tilde{v}^*)\|$(-)')
axs[0].set_title('Formación con n=4 agentes')
axs[0].set_xlabel('Tiempo (-)')
axs[0].set_xlim([0,100])
axs[0].grid(True)
axs[0].legend(loc='upper right', prop={'size': 12})

axs[1].plot(tiempos_2, errores_2, label=r'$\|z(t) - \tilde{z}^*\|$(-)')
axs[1].plot(tiempos_2, errores_velocidad_2, label=r'$\|\dot{p}(t) - (\mathbf{1} \otimes \tilde{v}^*)\|$(-)')
axs[1].set_title('Formación con n=15 agentes')
axs[1].set_xlabel('Tiempo (-)')
axs[1].set_xlim([0,15])
axs[1].grid(True)
axs[1].legend(loc='upper right', prop={'size': 12})

fig.suptitle(r'Convergerencia de las posiciones relativas a las posiciones relativas de referencia distorsionadas, y de las velocidades a la velocidad residual', fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

