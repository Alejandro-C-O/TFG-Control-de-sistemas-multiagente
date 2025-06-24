# -*- coding: utf-8 -*-
"""
Created on Sun May 11 19:04:41 2025

@author: jandr
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy import zeros, cos,sin
from matplotlib.patches import FancyArrowPatch

from matplotlib import cm
import matplotlib.gridspec as gridspec




#para pintar ambas estructuras en la misma figura


# -------- Subfigura 1: Grafo cuadrado --------
n1 = 4
ones_n = np.ones(n1)
p_cm = np.array([0, 0])
p_c_ref = np.array([ 
    [1], [1],    # Agente 1
    [-1], [1],   # Agente 2
    [-1], [-1],  # Agente 3
    [1], [-1]    # Agente 4
])
p_ref1 = p_c_ref
x1 = p_ref1[0::2, 0]
y1 = p_ref1[1::2, 0]

# -------- Subfigura 2: Grafo espiral --------
n2 = 15
a = 0.5
b = 0.5
theta = np.linspace(0, 4 * np.pi, n2)
r = a + b * theta
x2 = r * np.cos(theta)
y2 = r * np.sin(theta)

p_ref2 = np.empty((2 * n2, 1))
p_ref2[0::2, 0] = x2
p_ref2[1::2, 0] = y2

# -------- Figura combinada con subplots --------
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Subplot 1: Cuadrado
axs[0].plot(x1, y1, 'o', markersize=6, label='Agentes')
for i in range(n1):
    axs[0].text(x1[i] + 0.15, y1[i] + 0.05, str(i + 1), fontsize=10)
    # Conectar nodos en cadena
for i in range(n1 - 1):
    axs[0].plot([x1[i], x1[i+1]], [y1[i], y1[i+1]], 'k-')  # línea negra

axs[0].set_aspect('equal')
axs[0].set_xlim(-2, 2)
axs[0].set_ylim(-2, 2)
axs[0].set_xlabel('Posición X(-)')
axs[0].set_ylabel('Posición Y(-)')
axs[0].set_title("Formación de referencia cuadrado")
axs[0].grid(True)
axs[0].legend()

# Subplot 2: Espiral
axs[1].plot(x2, y2, 'o', markersize=6, label='Agentes')
for i in range(n2):
    axs[1].text(x2[i] + 0.15, y2[i] + 0.05, str(i + 1), fontsize=10)
# Conectar nodos en cadena
for i in range(n2 - 1):
    axs[1].plot([x2[i], x2[i+1]], [y2[i], y2[i+1]], 'k-')  # línea negra

axs[1].set_aspect('equal')
axs[1].set_xlabel('Posición X(-)')
axs[1].set_ylabel('Posición Y(-)')
axs[1].set_title("Formación de referencia en espiral")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()
















n = 15  # número de agentes
a = 0.5  # radio inicial
b = 0.5  # separación entre vueltas
theta = np.linspace(0, 4 * np.pi, n)  # ángulos equiespaciados

r = a + b * theta
x = r * np.cos(theta)
y = r * np.sin(theta)

# Vector columna p_ref (ordenado como [x1, y1, x2, y2, ..., xn, yn])
p_ref = np.empty((2*n, 1))
p_ref[0::2, 0] = x
p_ref[1::2, 0] = y


# Crear figura
plt.figure(figsize=(6, 6))
plt.plot(x, y, 'o', markersize=6, label='Agentes')

# Etiquetas de nodos (más pegadas)
for i in range(n):
    plt.text(x[i] + 0.15, y[i] + 0.05, str(i + 1), fontsize=10)

# Dibujar flechas entre los agentes 1→2→...→15
for i in range(n - 1):
    dx = x[i + 1] - x[i]
    dy = y[i + 1] - y[i]
    plt.arrow(x[i], y[i], dx, dy,  # Escalado para mejor visualización
              head_width=0.3, head_length=0.4, fc='k', ec='k', length_includes_head=True)

# Ajustes de visualización
plt.gca().set_aspect('equal')
plt.xlabel('Posición X(-)')
plt.ylabel('Posición Y(-)')
plt.grid(True)
plt.title("Grafo en espiral")
plt.legend()
plt.show()


# # Crear figura
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.plot(x, y, 'o-', label='Formación en espiral')

# # Etiquetas con número de agente
# for i in range(n):
#     ax.text(x[i] + 0.1, y[i] + 0.1, str(i + 1))

# # Dibujar flechas del i al i+1
# for i in range(n - 1):
#     arrow = FancyArrowPatch(
#         (x[i], y[i]), (x[i + 1], y[i + 1]),
#         arrowstyle='->', mutation_scale=30, color='gray', linewidth=1
#     )
#     ax.add_patch(arrow)

# # Ajustes del gráfico
# ax.set_aspect('equal')
# ax.grid(True)
# ax.set_title(r"Formación de referencia en espiral ($p^*$)")
# ax.legend()
# plt.show()


# # Graficamos la formación
# plt.figure(figsize=(6,6))
# plt.plot(x, y, 'o-', label='Formación en espiral')
# for i in range(n):
#     plt.text(x[i] + 0.1, y[i] + 0.1, str(i+1))
# plt.gca().set_aspect('equal')
# plt.grid(True)
# plt.title("Formación inicial en espiral (p_ref)")
# plt.legend()
# plt.show()


#cada sección tiene su dinámica correspondiente 

#III

aristas = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15)]

# Inicializar la matriz B con ceros
B = np.zeros((n, len(aristas)))

# Rellenar la matriz B según la ecuación (2)
for k, (tail,head) in enumerate(aristas):
    B[tail-1, k] = 1
    B[head-1, k] = -1

# Definir los pesos de las aristas (asumimos que todos los pesos son 1)
pesos = np.ones(len(aristas))

# Crear la matriz diagonal D_w
D_w = np.diag(pesos)

# Calcular la matriz Laplaciana L
L = B @ D_w @ B.T #ecuacion 3 artículo

np.random.seed(2)
# Definir un rango de valores para las posiciones iniciales
rango_x = (-10, 10)  # Rango de valores para las coordenadas x
rango_y = (-10, 10)  # Rango de valores para las coordenadas y

# Generar posiciones aleatorias para los agentes
x_init = np.random.uniform(rango_x[0], rango_x[1], size=n)
y_init = np.random.uniform(rango_y[0], rango_y[1], size=n)

# Vector columna p con las posiciones iniciales aleatorias
p = np.empty((2 * n, 1))
p[0::2, 0] = x_init
p[1::2, 0] = y_init


Bbarra=np.kron(B,np.eye(2))


z=Bbarra.T@p

Lbarra=np.kron(L,np.eye(2))



# # Parámetros del sistema
dt = 0.01
T= 600
traj = [p.copy()]
vel=[]

# Simulación de la dinámica con Euler
for t in range(int(T / dt)):
    pdot = -np.kron(L, np.eye(2)) @ (p - p_ref)  # Dinámica
    vel.append(pdot.copy())
    p = p + pdot * dt  # Actualización de las posiciones con el método de Euler
    traj.append(p.copy())  # Guardar las posiciones para graficar

# Graficar la trayectoria de los agentes
traj = np.array(traj)
vel=np.array(vel)
# Generar una lista de colores únicos para cada agente
colors = cm.jet(np.linspace(0, 1, n))  # 'jet' es un mapa de colores, pero puedes usar otros como 'viridis'

# Crear la figura
plt.figure(figsize=(8, 6))

# Graficar las trayectorias de los agentes
for i in range(n):
    plt.plot(traj[:, 2 * i], traj[:, 2 * i + 1], color=colors[i],linewidth=0.3)
    plt.scatter(traj[0, 2 * i], traj[0, 2 * i + 1], color=colors[i], marker='o')
    plt.scatter(traj[-1, 2 * i], traj[-1, 2 * i + 1], color=colors[i], marker='x')

# Marcar la posición de referencia
plt.plot(p_ref[::2], p_ref[1::2], 'ms',markersize=3, label='Referencia')

# Añadir las líneas discontinuas negras que unen las posiciones finales de los agentes
for i in range(n - 1):
    plt.plot([traj[-1, 2 * i], traj[-1, 2 * (i + 1)]], [traj[-1, 2 * i + 1], traj[-1, 2 * (i + 1) + 1]], 'k--', linewidth=1)

# Añadir las líneas discontinuas rojas que unen las posiciones de referencia
for i in range(n - 1):
    plt.plot([p_ref[2 * i, 0], p_ref[2 * (i + 1), 0]], [p_ref[2 * i + 1, 0], p_ref[2 * (i + 1) + 1, 0]], 'm--', linewidth=1)


plt.scatter([], [], color='k', marker='o', label='Posición inicial')
plt.scatter([], [], color='k', marker='x', label='Posición final')

# Establecer etiquetas y título
plt.xlabel('Posición X(-)')
plt.ylabel('Posición Y(-)')
plt.title('Trayectoria de los agentes')
plt.legend()

plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')

# Mostrar el gráfico sin leyenda
plt.show()




#añadir también la evolucion de las velocidades a 0
# Crear la figura general
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])

# Subplot 1 (arriba a la derecha): velocidades en X
ax1 = fig.add_subplot(gs[0, 1])
for i in range(n):
    ax1.plot(vel[:, 2 * i], color=colors[i])
ax1.set_title('Velocidades en X(-)')
ax1.set_xlabel('Tiempo de simulación(-)')
ax1.set_ylabel('Velocidad(-)')
ax1.set_xlim([0, 600])
ax1.set_ylim([-10, 10])
ax1.grid(True)
ax1.legend()

# Subplot 2 (abajo a la derecha): velocidades en Y
ax2 = fig.add_subplot(gs[1, 1])
for i in range(n):
    ax2.plot(vel[:, 2 * i + 1], color=colors[i])
ax2.set_title('Velocidades en Y(-)')
ax2.set_xlabel('Tiempo de simulación(-)')
ax2.set_ylabel('Velocidad(-)')
ax2.set_xlim([0, 600])
ax2.set_ylim([-10, 10])
ax2.grid(True)
ax2.legend()

# Subplot 3 (izquierda, ocupa ambas filas): trayectoria
ax3 = fig.add_subplot(gs[:, 0])
for i in range(n):
    ax3.plot(traj[:, 2 * i], traj[:, 2 * i + 1], color=colors[i], linewidth=0.3)
    ax3.scatter(traj[0, 2 * i], traj[0, 2 * i + 1], color=colors[i], marker='o')
    ax3.scatter(traj[-1, 2 * i], traj[-1, 2 * i + 1], color=colors[i], marker='x')

# Marcar la posición de referencia
ax3.plot(p_ref[::2], p_ref[1::2], 'ms', markersize=3, label='Referencia')

# Líneas negras entre posiciones finales
for i in range(n - 1):
    ax3.plot([traj[-1, 2 * i], traj[-1, 2 * (i + 1)]],
             [traj[-1, 2 * i + 1], traj[-1, 2 * (i + 1) + 1]], 'k--', linewidth=1)

# Líneas magenta entre posiciones de referencia
for i in range(n - 1):
    ax3.plot([p_ref[2 * i, 0], p_ref[2 * (i + 1), 0]],
             [p_ref[2 * i + 1, 0], p_ref[2 * (i + 1) + 1, 0]], 'm--', linewidth=1)

ax3.scatter([], [], color='k', marker='o', label='Posición inicial')
ax3.scatter([], [], color='k', marker='x', label='Posición final')
ax3.set_xlabel('Posición X (-)')
ax3.set_ylabel('Posición Y (-)')
ax3.set_title('Trayectoria de los agentes, tiempo de integración = 600 (-)')
ax3.grid(True)
ax3.legend()
ax3.set_aspect('equal', adjustable='box')

# Ajustes finales
plt.suptitle('Trayectoria y convergencia de velocidades de los agentes')
plt.tight_layout()
plt.show()










#me he "saltado" las dos secciones anteriores aqui, para 4 tengo todo perfectamente definido
#preguntar qué es lo que hace falta poner exactamente
#seccion IV C

# Parámetros
n = 15
m = 2
kappa = 0.08
v_star_i = np.array([[1], [1.6]])

# Topología en línea
edges = [(i, i+1) for i in range(n-1)]

# Espiral
a = 0.5
b = 0.5
theta = np.linspace(0, 4 * np.pi, n)
r = a + b * theta
x = r * np.cos(theta)
y = r * np.sin(theta)

# Vector columna p_ref (2n x 1)
p_ref = np.empty((2*n, 1))
p_ref[0::2, 0] = x
p_ref[1::2, 0] = y
p_i_ref = p_ref.reshape((n, m))

# Construcción de vecinos bidireccionales
vecinos_dict = {i: [] for i in range(n)}
for i, j in edges:
    vecinos_dict[i].append(j)
    vecinos_dict[j].append(i)

# Inicializar mu_{ij}
mu_matrices = np.zeros((n, n, 2, 2))

# Cálculo de mu_{ij} para cada agente (versión corregida)
for i in range(n):
    vecinos = vecinos_dict[i]
    num_vecinos = len(vecinos)

    A = np.zeros((2, 2 * num_vecinos))

    for k, j in enumerate(vecinos):
        delta = p_i_ref[i] - p_i_ref[j]
        dx, dy = delta.flatten()

        A[0, 2*k]     = dx
        A[0, 2*k + 1] = -dy
        A[1, 2*k]     = dy
        A[1, 2*k + 1] = dx

    b = (v_star_i / kappa).flatten().reshape(-1, 1)

    X, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    for k, j in enumerate(vecinos):
        alpha = X[2 * k, 0]
        beta  = X[2 * k + 1, 0]
        mu_matrices[i, j] = np.array([[alpha, -beta], [beta, alpha]])

# Verificación
print("\nVerificación de la ecuación para cada agente:")
for i in range(n):
    suma = np.zeros(2)
    for j in range(n):
        if mu_matrices[i, j].any():
            diff = p_i_ref[i] - p_i_ref[j]
            suma += mu_matrices[i, j] @ diff
    v_i_estimada = kappa * suma
    error = np.linalg.norm(v_i_estimada - v_star_i.flatten())
    print(f"Agente {i+1}: v_i estimada = {v_i_estimada.round(4)}, error = {error:.6f}")

# Mostrar coeficientes distintos de cero
print("\nCoeficientes mu_{ij} distintos de cero:")
for i in range(n):
    for j in range(n):
        if mu_matrices[i, j].any():
            print(f"\nμ_{i+1},{j+1} =")
            print(mu_matrices[i, j].round(4))






dim=2
edges=np.array(aristas)
num_edges = edges.shape[0]


Msombrero = np.zeros((n * dim, num_edges * dim))

for k, (tail,head) in enumerate(edges):
    i = head - 1
    j = tail - 1

    # Bloques 2x2
    mu_ji = mu_matrices[j, i]  # μ_{tail, head}
    mu_ij = mu_matrices[i, j]  # μ_{head, tail}

    # Posiciones de fila y columna
    row_i = i * dim
    row_j = j * dim
    col = k * dim

    # Insertar bloques 2x2
    Msombrero[row_i:row_i+dim, col:col+dim] = -mu_ij
    Msombrero[row_j:row_j+dim, col:col+dim] = mu_ji



Lambdasombrero=Msombrero@Bbarra.T

pdot=-(Lbarra-kappa*Lambdasombrero)@p+Lbarra@p_ref




#seccion D

z=Bbarra.T@p



z_ref=Bbarra.T@p_ref


#we define the error signal
e=z-z_ref


#vamos a comprobar que todos los términos de la ecuacion 36 son iguales
segundo36=-np.kron(B@D_w@B.T,np.eye(2))@(p-p_ref)+kappa*Lambdasombrero@p

tercer36=-Bbarra@e+kappa*Msombrero@z

cuarto36=-Bbarra@e+kappa*Msombrero@e+kappa*Msombrero@z_ref


v_star=v_star_i
quinto36=-Bbarra@e+kappa*Msombrero@e+np.kron(np.ones((n,1)),v_star)

#compruebo y comento
# print(f"primer termino36 ,pdot={pdot}")
# print("segundo termino 36=",segundo36)
# print("tercer termino 36=",tercer36)
# print("cuarto termino 36=",cuarto36)
# print("quinto termino 36=",quinto36)



#comprobación ecuacion 37 y la igualdad de arriba
zdot=Bbarra.T@pdot

edot=zdot


segundo37=-Bbarra.T@Bbarra@e+kappa*Bbarra.T@Msombrero@e+kappa*Bbarra.T@np.kron(np.ones((n,1)),v_star)

tercero37=-Bbarra.T@Bbarra@e+kappa*Bbarra.T@Msombrero@e


#compruebo y comento
# print(np.round(np.hstack([
#     Bbarra.T @ pdot,
#     zdot,
#     edot,
#     segundo37,
#     tercero37
# ]), decimals=4))





#dinámica con mu_ij matrices, plot->

np.random.seed(1)
# Definir un rango de valores para las posiciones iniciales
rango_x = (-10, 10)  # Rango de valores para las coordenadas x
rango_y = (-10, 10)  # Rango de valores para las coordenadas y

# Generar posiciones aleatorias para los agentes
x_init = np.random.uniform(rango_x[0], rango_x[1], size=n)
y_init = np.random.uniform(rango_y[0], rango_y[1], size=n)

# Vector columna p con las posiciones iniciales aleatorias
p = np.empty((2 * n, 1))
p[0::2, 0] = x_init
p[1::2, 0] = y_init


# # Parámetros del sistema
dt = 0.01
T= 40
traj = [p.copy()]
vel=[]


# Simulación de la dinámica con Euler
for t in range(int(T / dt)):
    pdot = -(Lbarra-kappa*Lambdasombrero)@p+Lbarra@p_ref   # Dinámica
    vel.append(pdot.copy())
    p = p + pdot * dt  # Actualización de las posiciones con el método de Euler
    traj.append(p.copy())  # Guardar las posiciones para graficar

# Graficar la trayectoria de los agentes
traj = np.array(traj)
vel=np.array(vel)

# Generar una lista de colores únicos para cada agente
colors = cm.jet(np.linspace(0, 1, n))  # 'jet' es un mapa de colores, pero puedes usar otros como 'viridis'

# Crear la figura
plt.figure(figsize=(8, 6))

# Graficar las trayectorias de los agentes
for i in range(n):
    plt.plot(traj[:, 2 * i], traj[:, 2 * i + 1], color=colors[i],linewidth=0.3)
    plt.scatter(traj[0, 2 * i], traj[0, 2 * i + 1], color=colors[i], marker='o')
    plt.scatter(traj[-1, 2 * i], traj[-1, 2 * i + 1], color=colors[i], marker='x')

# Marcar la posición de referencia
plt.plot(p_ref[::2], p_ref[1::2], 'ms',markersize=3,label='Referencia')

# Añadir las líneas discontinuas negras que unen las posiciones finales de los agentes
for i in range(n - 1):
    plt.plot([traj[-1, 2 * i], traj[-1, 2 * (i + 1)]], [traj[-1, 2 * i + 1], traj[-1, 2 * (i + 1) + 1]], 'k--', linewidth=1)

# Añadir las líneas discontinuas rojas que unen las posiciones de referencia
for i in range(n - 1):
    plt.plot([p_ref[2 * i, 0], p_ref[2 * (i + 1), 0]], [p_ref[2 * i + 1, 0], p_ref[2 * (i + 1) + 1, 0]], 'm--', linewidth=1)

plt.scatter([], [], color='k', marker='o', label='Posición inicial')
plt.scatter([], [], color='k', marker='x', label='Posición final')

# Establecer etiquetas y título
plt.xlabel('Posición X(-)')
plt.ylabel('Posición Y(-)')
plt.title('Trayectoria de los agentes, tiempo de integración=40(-)')
plt.grid(True)
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')

# Mostrar el gráfico sin leyenda
plt.show()




#velocidades
fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

# Subfigura 1: velocidades en X
for i in range(n):
    axs[0].plot(vel[:, 2 * i],  color=colors[i]) #label=f'Agente {i+1}',
axs[0].set_title('Velocidades en X')
axs[0].set_xlabel('Tiempo de simulación(-)')
axs[0].set_ylabel('Velocidad')
axs[0].set_xlim([0,1000])
axs[0].set_ylim([-25,25])
axs[0].grid(True)
axs[0].legend()

# Subfigura 2: velocidades en Y
for i in range(n):
    axs[1].plot(vel[:, 2 * i + 1],  color=colors[i]) #label=f'Agente {i+1}',
axs[1].set_title('Velocidades en Y')
axs[0].set_ylim([-0.5,1])
axs[1].set_xlabel('Tiempo de simulación(-)')
axs[0].set_xlim([0,1000])
axs[0].set_ylim([-20,20])
axs[1].grid(True)
axs[1].legend()

plt.suptitle('Convergencia de las velocidades de los agentes')
plt.tight_layout()
plt.show()



#pintando todo lo de IV.C todo junto
# Crear la figura general
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])

# Subplot 1 (arriba a la derecha): velocidades en X
ax1 = fig.add_subplot(gs[0, 1])
for i in range(n):
    ax1.plot(vel[:, 2 * i], color=colors[i])
ax1.set_title('Velocidades en X(-)')
ax1.set_xlabel('Tiempo de simulación(-)')
ax1.set_ylabel('Velocidad')
ax1.set_xlim([0, 600])
ax1.set_ylim([-10, 10])
ax1.grid(True)
ax1.legend()

# Subplot 2 (abajo a la derecha): velocidades en Y
ax2 = fig.add_subplot(gs[1, 1])
for i in range(n):
    ax2.plot(vel[:, 2 * i + 1], color=colors[i])
ax2.set_title('Velocidades en Y(-)')
ax2.set_xlabel('Tiempo de simulación(-)')
ax2.set_ylabel('Velocidad')
ax2.set_xlim([0, 600])
ax2.set_ylim([-10, 10])
ax2.grid(True)
ax2.legend()

# Subplot 3 (izquierda, ocupa ambas filas): trayectoria
ax3 = fig.add_subplot(gs[:, 0])
for i in range(n):
    ax3.plot(traj[:, 2 * i], traj[:, 2 * i + 1], color=colors[i], linewidth=0.3)
    ax3.scatter(traj[0, 2 * i], traj[0, 2 * i + 1], color=colors[i], marker='o')
    ax3.scatter(traj[-1, 2 * i], traj[-1, 2 * i + 1], color=colors[i], marker='x')

# Marcar la posición de referencia
ax3.plot(p_ref[::2], p_ref[1::2], 'ms', markersize=3, label='Referencia')

# Líneas negras entre posiciones finales
for i in range(n - 1):
    ax3.plot([traj[-1, 2 * i], traj[-1, 2 * (i + 1)]],
             [traj[-1, 2 * i + 1], traj[-1, 2 * (i + 1) + 1]], 'k--', linewidth=1)

# Líneas magenta entre posiciones de referencia
for i in range(n - 1):
    ax3.plot([p_ref[2 * i, 0], p_ref[2 * (i + 1), 0]],
             [p_ref[2 * i + 1, 0], p_ref[2 * (i + 1) + 1, 0]], 'm--', linewidth=1)

ax3.scatter([], [], color='k', marker='o', label='Posición inicial')
ax3.scatter([], [], color='k', marker='x', label='Posición final')
ax3.set_xlabel('Posición X (-)')
ax3.set_ylabel('Posición Y (-)')
ax3.set_title('Trayectoria de los agentes, tiempo de integración = 40 (-)')
ax3.grid(True)
ax3.legend()
ax3.set_aspect('equal', adjustable='box')

# Ajustes finales
plt.suptitle('Trayectoria y convergencia de velocidades de los agentes')
plt.tight_layout()
plt.show()



#seccion V metemos imperfecciones en las medidas

n=15 #agentes
m=2 #dimensiones


a=[1.02,1,0.98,1,1,1.04,1,.96,1,1,1.03,.97,1,1.01,.99]

#necesitamos hacer unas cuantas cosas para calcular D_R
theta=[0.06,0,-0.06,-0.09,0,-.03,.02,-.04,0,.01,0,0,-.02,0,.03]

#voy a probar a poner a y theta de forma que sean más cercanas a lo ideal a ver si 
#esto hace que ptilde_ref me salga más cercana a p_ref

# a=[1.04,0.98,1.02,.97]
# theta=[0.03,-.05,.04,-.01]


#como es totalmente lógico y esperado,
#veo que cuanto más acerquemos estos parámetros a los ideales, más pequeña es el 
#movimiento no deseado, es decir, menor vtilde_ref  
#también mejor ptilde_ref pero en la velocidad se ve todavía más claramente



listaR=[]
listaR_RiT=[]

for i in range(n):
    R_i=np.array([[cos(theta[i]),sin(theta[i])],[-sin(theta[i]),cos(theta[i])]])
    listaR.append(R_i)
    #print("R_",i+1,".T=",R_i.T)
    listaR_RiT.append(R_i.T)

  
R=np.hstack(listaR_RiT).T


def bloque_diagonal(lista_de_matrices):
    n = len(lista_de_matrices)
    m = lista_de_matrices[0].shape[0]
    
    return np.block([
        [lista_de_matrices[i] if i == j else np.zeros((m, m)) for j in range(n)]
        for i in range(n)
    ])
    
D_R=bloque_diagonal(listaR)



#lo puedo hacer así, pero al final es más fácil teniendo 
#a como una lista, hacerlo como se hace después del comentario

# a=[np.array([[1.1]]),np.array([[.75]]),np.array([[.93]]),np.array([[1.34]])]
# Da=bloque_diagonal(a)


Da=np.diag(a)

Dabarra=np.kron(Da,np.eye(2))
Dx=Dabarra@D_R


pdot=-Dx@Lbarra@p+Lbarra@p_ref




np.random.seed(0)
# Definir un rango de valores para las posiciones iniciales
rango_x = (-10, 10)  # Rango de valores para las coordenadas x
rango_y = (-10, 10)  # Rango de valores para las coordenadas y

# Generar posiciones aleatorias para los agentes
x_init = np.random.uniform(rango_x[0], rango_x[1], size=n)
y_init = np.random.uniform(rango_y[0], rango_y[1], size=n)

# Vector columna p con las posiciones iniciales aleatorias
p = np.empty((2 * n, 1))
p[0::2, 0] = x_init
p[1::2, 0] = y_init
# # Parámetros del sistema
dt = 0.01
T= 1200
traj = [p.copy()]
pdots=[]
# Simulación de la dinámica con Euler
for t in range(int(T / dt)):
    pdot = -Dx@Lbarra@p+Lbarra@p_ref   # Dinámica
    pdots.append(pdot.copy())
    p = p + pdot * dt  # Actualización de las posiciones con el método de Euler
    traj.append(p.copy())  # Guardar las posiciones para graficar

# Graficar la trayectoria de los agentes
traj = np.array(traj)
pdots=np.array(pdots)

# Generar una lista de colores únicos para cada agente
colors = cm.jet(np.linspace(0, 1, n))  # 'jet' es un mapa de colores, pero puedes usar otros como 'viridis'

# Crear la figura
plt.figure(figsize=(8, 6))

# Graficar las trayectorias de los agentes
for i in range(n):
    plt.plot(traj[:, 2 * i], traj[:, 2 * i + 1], color=colors[i],linewidth=0.3)
    plt.scatter(traj[0, 2 * i], traj[0, 2 * i + 1], color=colors[i], marker='o')
    plt.scatter(traj[-1, 2 * i], traj[-1, 2 * i + 1], color=colors[i], marker='x')

# Marcar la posición de referencia
plt.plot(p_ref[::2], p_ref[1::2], 'ms',markersize=3,label='Referencia')

# Añadir las líneas discontinuas negras que unen las posiciones finales de los agentes
for i in range(n - 1):
    plt.plot([traj[-1, 2 * i], traj[-1, 2 * (i + 1)]], [traj[-1, 2 * i + 1], traj[-1, 2 * (i + 1) + 1]], 'k--', linewidth=1)

# Añadir las líneas discontinuas rojas que unen las posiciones de referencia
for i in range(n - 1):
    plt.plot([p_ref[2 * i, 0], p_ref[2 * (i + 1), 0]], [p_ref[2 * i + 1, 0], p_ref[2 * (i + 1) + 1, 0]], 'm--', linewidth=1)

plt.scatter([], [], color='k', marker='o', label='Posición inicial')
plt.scatter([], [], color='k', marker='x', label='Posición final')
# Establecer etiquetas y título
plt.xlabel('Posición X(-)')
plt.ylabel('Posición Y(-)')
plt.title('Trayectoria de los agentes, tiempo de integración=1200(-)')
plt.grid(True)
plt.legend(loc='upper left', bbox_to_anchor=(-.01, .97),borderpad=.1)
plt.gca().set_aspect('equal', adjustable='box')

# Mostrar el gráfico sin leyenda
plt.show()


#para pintar a la vez trayectorias y evolución de las velocidades
# Crear la figura general
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])

# Subplot 1 (arriba a la derecha): velocidades en X
ax1 = fig.add_subplot(gs[0, 1])
for i in range(n):
    ax1.plot(pdots[:, 2 * i], color=colors[i])
ax1.set_title('Velocidades en X(-)')
ax1.set_xlabel('Tiempo de simulación(-)')
ax1.set_ylabel('Velocidad(-)')
ax1.set_xlim([0, 600])
ax1.set_ylim([-10, 10])
ax1.grid(True)
ax1.legend()

# Subplot 2 (abajo a la derecha): velocidades en Y
ax2 = fig.add_subplot(gs[1, 1])
for i in range(n):
    ax2.plot(pdots[:, 2 * i + 1], color=colors[i])
ax2.set_title('Velocidades en Y(-)')
ax2.set_xlabel('Tiempo de simulación(-)')
ax2.set_ylabel('Velocidad(-)')
ax2.set_xlim([0, 600])
ax2.set_ylim([-10, 10])
ax2.grid(True)
ax2.legend()

# Subplot 3 (izquierda, ocupa ambas filas): trayectoria
ax3 = fig.add_subplot(gs[:, 0])
for i in range(n):
    ax3.plot(traj[:, 2 * i], traj[:, 2 * i + 1], color=colors[i], linewidth=0.3)
    ax3.scatter(traj[0, 2 * i], traj[0, 2 * i + 1], color=colors[i], marker='o')
    ax3.scatter(traj[-1, 2 * i], traj[-1, 2 * i + 1], color=colors[i], marker='x')

# Marcar la posición de referencia
ax3.plot(p_ref[::2], p_ref[1::2], 'ms', markersize=3, label='Referencia')

# Líneas negras entre posiciones finales
for i in range(n - 1):
    ax3.plot([traj[-1, 2 * i], traj[-1, 2 * (i + 1)]],
             [traj[-1, 2 * i + 1], traj[-1, 2 * (i + 1) + 1]], 'k--', linewidth=1)

# Líneas magenta entre posiciones de referencia
for i in range(n - 1):
    ax3.plot([p_ref[2 * i, 0], p_ref[2 * (i + 1), 0]],
             [p_ref[2 * i + 1, 0], p_ref[2 * (i + 1) + 1, 0]], 'm--', linewidth=1)

ax3.scatter([], [], color='k', marker='o', label='Posición inicial')
ax3.scatter([], [], color='k', marker='x', label='Posición final')
ax3.set_xlabel('Posición X (-)')
ax3.set_ylabel('Posición Y (-)')
ax3.set_title('Trayectoria de los agentes, tiempo de integración = 1200 (-)')
ax3.grid(True)
ax3.legend()
ax3.set_aspect('equal', adjustable='box')

# Ajustes finales
plt.suptitle('Trayectoria y convergencia de velocidades de los agentes')
plt.tight_layout()
plt.show()





#quiero calcular v˜* y p˜* las posiciones y velocidades distorsionadas

#calculo Mcuenca con 53 y resuelvo v˜* y p˜* de las ecuaciones de (46)

Mcuenca=-(Dx@Bbarra-Bbarra@np.linalg.inv(Bbarra.T@Bbarra)@Bbarra.T@Dx@Bbarra)@np.kron(D_w,np.eye(2))




#ecuacion 46 (despejo del primer término, pero no sé si la matriz va a ser invertible)

ptilde_ref=np.linalg.solve((Dx@np.kron(B@D_w,np.eye(2))+Mcuenca)@Bbarra.T,Lbarra@p_ref)


ztilde_ref=Bbarra.T@ptilde_ref

# print('z_ref=')
# print(z_ref)

# print('ztilde_ref=')
# print(ztilde_ref)


#poner matrices hasta donde se ha llegado y la referencia al lado para comparar

p_ref_mat = p_ref.reshape((n, m))
ptilde_ref_mat = ptilde_ref.reshape((n, m))

# Creamos matrices de diferencias relativas entre agentes consecutivos (i, i+1)
diffs_ref = p_ref_mat[1:] - p_ref_mat[:-1]
diffs_ptilde = ptilde_ref_mat[1:] - ptilde_ref_mat[:-1]

# Imprimir encabezado
print(f"{'ztilde_ref (relativas)'} {'pz_ref (relativas)'}")
print('-' * 62)

# Imprimir las diferencias fila por fila
for i in range(n - 1):
    dx1, dy1 = diffs_ptilde[i]
    dx2, dy2 = diffs_ref[i]
    print(f"[{dx1:8.4f}, {dy1:8.4f}]         [{dx2:8.4f}, {dy2:8.4f}]")





#voy a calcular v˜* 

vtilde_ref_full = Mcuenca @ Bbarra.T@ptilde_ref  # ∈ ℝ^{mn}
vtilde_ref_blocks = vtilde_ref_full.reshape((n, m))  # n filas, cada una es vtilde_i

# Calcular promedio (asume que todos tienen el mismo vtilde*)
vtilde_ref = np.mean(vtilde_ref_blocks, axis=0).reshape((m, 1)) #son todos iguales

print("vtilde_ref=")
print(vtilde_ref)


np.savez('datos_caso2.npz', traj=traj, pdots=pdots, ptilde_ref=ptilde_ref, Bbarra=Bbarra, vtilde_ref=vtilde_ref)

#al dibujarlo ptilde_ref, vemos que el ptilde_ref que se alcanza es bastante parecido a
#p_ref, más parecido cuando menores sean los errores introducidos



n = int(p_ref.shape[0] // 2)

p_ref_2D = p_ref.reshape((n, 2))
ptilde_ref_2D = ptilde_ref.reshape((n, 2))

#print(ptilde_ref_2D)

# Crear el gráfico
plt.figure(figsize=(6, 6))
plt.plot(p_ref_2D[:, 0], p_ref_2D[:, 1], 'bo-', label=r'Formación original de referencia, $p^*$')
plt.plot(ptilde_ref_2D[:, 0], ptilde_ref_2D[:, 1], 'ro--', label=r'Formación distorsionada alcanzada, $\tilde{p}^*$')

# Añadir etiquetas a cada agente
for i in range(n):
    plt.text(p_ref_2D[i, 0] + 0.05, p_ref_2D[i, 1] + 0.05, f'{i+1}', color='blue')
    plt.text(ptilde_ref_2D[i, 0] + 0.05, ptilde_ref_2D[i, 1] - 0.1, f'{i+1}', color='red')

plt.title(r'Comparación de formaciones: $p^*$ vs. $\tilde{p}^*$')
plt.xlabel('Posición X(-)')
plt.ylabel('Posición Y(-)')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()



# # Vector de diferencias relativas de referencia
# ztilde_ref = Bbarra.T @ ptilde_ref  # B^T ptilde^*

# # Cálculo del error ||z(t) - ztilde^*|| en cada instante
# errores = []
# for p_t in traj[0:4000]:
#     z_t = Bbarra.T @ p_t.reshape((-1, 1))  # B^T p(t)
#     error = np.linalg.norm(z_t - ztilde_ref)
#     errores.append(error)

# # Graficar la evolución del error
# plt.figure(figsize=(8, 5))
# plt.plot(np.arange(len(errores)) * dt, errores, label=r'$\|z(t) - \tilde{z}^*\|$')
# plt.xlabel('Tiempo (-)')
# plt.title(r'Convergencia de $z(t)$ hacia $\tilde{z}^*$')
# plt.grid(True)
# plt.legend(prop={'size': 14})  
# plt.tight_layout()
# plt.show()


# # Vector de diferencias relativas de referencia
# ztilde_ref = Bbarra.T @ ptilde_ref  # B^T ptilde^*

# # Cálculo del error ||z(t) - ztilde^*|| en cada instante
# errores = []
# for p_t in traj:
#     z_t = Bbarra.T @ p_t.reshape((-1, 1))  # B^T p(t)
#     error = np.linalg.norm(z_t - ztilde_ref)
#     errores.append(error)

# vtilde_ref_global = np.kron(np.ones((n, 1)), vtilde_ref)
# errores_velocidad = []
# for pdot_t in pdots:
#     error_v = np.linalg.norm(pdot_t - vtilde_ref_global)
#     errores_velocidad.append(error_v)

# errores = errores[4001:]
# errores_velocidad = errores_velocidad[4000:]
# tiempos = np.arange(4000, 4000 + len(errores)) * dt

# plt.figure(figsize=(9, 5))

# plt.plot(tiempos, errores, label=r'$\|z(t) - \tilde{z}^*\|$')
# plt.plot(tiempos, errores_velocidad, label=r'$\|\dot{p}(t) - (\mathbf{1} \otimes \tilde{v}^*)\|$')

# plt.xlabel('Tiempo (-)')
# plt.title(r'Convergencia de $z(t)$ y $\dot{p}(t)$')
# plt.grid(True)
# plt.legend(prop={'size': 14})
# plt.tight_layout()
# plt.show()





# # Vector de diferencias relativas de referencia
# ztilde_ref = Bbarra.T @ ptilde_ref  # B^T ptilde^*

# # Cálculo del error ||z(t) - ztilde^*|| en cada instante
# errores = []
# for p_t in traj:
#     z_t = Bbarra.T @ p_t.reshape((-1, 1))  # B^T p(t)
#     error = np.linalg.norm(z_t - ztilde_ref)
#     errores.append(error)

# vtilde_ref_global = np.kron(np.ones((n, 1)), vtilde_ref)
# errores_velocidad = []
# for pdot_t in pdots:
#     error_v = np.linalg.norm(pdot_t - vtilde_ref_global)
#     errores_velocidad.append(error_v)

# errores = errores[1:]
# tiempos = np.arange(len(errores)) * dt

# plt.figure(figsize=(9, 5))
# tiempos = np.arange(len(errores)) * dt
# plt.plot(tiempos, errores, label=r'$\|z(t) - \tilde{z}^*\|$')
# plt.plot(tiempos, errores_velocidad, label=r'$\|\dot{p}(t) - (\mathbf{1} \otimes \tilde{v}^*)\|$')

# plt.xlabel('Tiempo (-)')
# plt.title(r'Convergencia de $z(t)$ y $\dot{p}(t)$')
# plt.grid(True)
# plt.legend(prop={'size': 14})
# plt.tight_layout()
# plt.show()


# Vector de diferencias relativas de referencia
ztilde_ref = Bbarra.T @ ptilde_ref  # B^T ptilde^*

# Cálculo del error ||z(t) - ztilde^*|| en cada instante
errores = []
for p_t in traj:
    z_t = Bbarra.T @ p_t.reshape((-1, 1))  # B^T p(t)
    error = np.linalg.norm(z_t - ztilde_ref)
    errores.append(error)

vtilde_ref_global = np.kron(np.ones((n, 1)), vtilde_ref)
errores_velocidad = []
for pdot_t in pdots:
    error_v = np.linalg.norm(pdot_t - vtilde_ref_global)
    errores_velocidad.append(error_v)

errores = errores[:4000]
errores_velocidad=errores_velocidad[:4000]
tiempos = np.arange(len(errores)) * dt

plt.figure(figsize=(9, 5))
tiempos = np.arange(len(errores)) * dt
plt.plot(tiempos, errores, label=r'$\|z(t) - \tilde{z}^*\|$')
plt.plot(tiempos, errores_velocidad, label=r'$\|\dot{p}(t) - (\mathbf{1} \otimes \tilde{v}^*)\|$')

plt.xlabel('Tiempo (-)')
plt.title(r'Convergencia de $z(t)$ y $\dot{p}(t)$')
plt.grid(True)
plt.legend(prop={'size': 14})
plt.tight_layout()
plt.show()
