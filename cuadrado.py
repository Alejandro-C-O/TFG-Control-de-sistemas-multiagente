# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 13:02:11 2025

@author: jandr
"""

#hay un error en la notacion de la teoria de grafos en el artículo de hector
#primero se ha hecho el codigo siguiente (head, tail) como pone en la ecuacion 
#pero luego pone que el primer termino es el tail y el segundo el head, y esto es
#lo consistente con la literatura.
# por tanto hay que cambiar el codigo para que sea (tail,head) y no al revés



#scipy para los solvers
#python ode solvers

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from scipy.integrate import solve_ivp
from numpy import zeros,cos,sin
from sympy import solve
from matplotlib import cm
import matplotlib.gridspec as gridspec

# Definir los nodos y las aristas
nodos = 4
aristas = [(1, 2), (2, 3), (3, 4)]

# Inicializar la matriz B con ceros
B = np.zeros((nodos, len(aristas)))

# Rellenar la matriz B según la ecuación (2)
for k, (tail,head) in enumerate(aristas):
    B[tail-1, k] = 1
    B[head-1, k] = -1

print("Matriz de Incidencia B:")
print(B)

# Definir los pesos de las aristas (asumimos que todos los pesos son 1)
pesos = np.ones(len(aristas))

# Crear la matriz diagonal D_w
D_w = np.diag(pesos)

# Calcular la matriz Laplaciana L
L = B @ D_w @ B.T #ecuacion 3 artículo


#vemos que calculando L según la ecaución 1 obtenemos lo mismo
def crear_matriz_laplaciana(num_nodos, vecinos, pesos):
    # Inicializar la matriz laplaciana de tamaño num_nodos x num_nodos
    L = np.zeros((num_nodos, num_nodos))
    
    for i in range(num_nodos):
        # Sumar los pesos de los vecinos de i (para la diagonal)
        L[i, i] = sum(pesos.get((i, k), 0) for k in vecinos[i])
        
        for j in vecinos[i]:
            if i != j:  # Si no estamos en la diagonal
                # Si i y j están conectados, poner el peso negativo
                if (i, j) in pesos:
                    L[i, j] = -pesos[(i, j)]
                elif (j, i) in pesos:
                    L[i, j] = -pesos[(j, i)]
    
    return L

num_nodos = 4

vecinos = [
    [1],       # Nodo 0 (nodo 1) tiene como vecino al nodo 1
    [0, 2],    # Nodo 1 (nodo 2) tiene como vecinos a los nodos 0 y 2
    [1, 3],    # Nodo 2 (nodo 3) tiene como vecinos a los nodos 1 y 3
    [2]        # Nodo 3 (nodo 4) tiene como vecino al nodo 2
]

pesos_ = {
    (0, 1): 1.0, (1, 0): 1.0,  # Arista (1, 2) bidireccional
    (1, 2): 1.0, (2, 1): 1.0,  # Arista (2, 3) bidireccional
    (2, 3): 1.0, (3, 2): 1.0   # Arista (3, 4) bidireccional
}


print("pesos= ? w_ij? =")
print(pesos_)
Lec1= crear_matriz_laplaciana(num_nodos, vecinos, pesos_)


print("Laplaciana por definición")
print(Lec1)

print("Matriz de Incidencia B:")
print(B)
print("\nMatriz Diagonal D_w:")
print(D_w)
print("\nMatriz Laplaciana L:")
print(L)

# Verificar la simetría
es_simetrica = np.allclose(L, L.T)

# Calcular los valores propios
valores_propios = np.linalg.eigvals(L)

# Verificar que todos los valores propios son no negativos
valores_propios_no_negativos = np.all(valores_propios >= 0)

print("Matriz Laplaciana L:")
print(L)
print("\n¿Es simétrica?:", es_simetrica)
print("Valores propios:", valores_propios)
print("¿Todos los valores propios son no negativos?:", valores_propios_no_negativos)


# Número de agentes
n = 4

# Vector de unos de tamaño n
ones_n = np.ones(n)

# Posición del centro de masa de la configuración de referencia (asumimos que es el origen para simplificar)
p_cm = np.array([0, 0])

# Vector de posiciones relativas desde el centro de masa de la configuración de referencia
p_c_ref = np.array([ 
    [1], [1],  # Agente 1
    [-1], [1],  # Agente 2
    [-1], [-1],  # Agente 3
    [1], [-1]  # Agente 4
])

# Calcular el vector de configuración de referencia p* utilizando el producto de Kronecker
#p_ref = np.kron(ones_n, p_cm) + p_c_ref #ecuacion 4 artículo

p_ref=p_c_ref
print("Vector de configuración de referencia p*:")
print(p_ref)



x = p_ref[0::2, 0]
y = p_ref[1::2, 0]

# Crear figura
plt.figure(figsize=(6, 6))
plt.plot(x, y, 'o', markersize=6, label='Agentes')

# Etiquetas de nodos (más pegadas)
for i in range(n):
    plt.text(x[i] + 0.15, y[i] + 0.05, str(i + 1), fontsize=10)

# Dibujar flechas entre los agentes 1→2→3→4
for i in range(n - 1):
    dx = x[i + 1] - x[i]
    dy = y[i + 1] - y[i]
    plt.arrow(x[i], y[i], dx , dy ,  # Acortar un poco la flecha
              head_width=0.1, head_length=0.15, fc='k', ec='k', length_includes_head=True)

# Ajustes de visualización
plt.gca().set_aspect('equal')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xlabel('Posición X(-)')
plt.ylabel('Posición Y(-)')
plt.grid(True)
plt.title("Grafo cuadrado")
plt.legend()
plt.show()


# Vector de traslación b (asumimos una traslación arbitraria)
b = np.array([2, 6])

# Calcular el conjunto de configuraciones admisibles S
#p = p_ref + np.kron(ones_n, b)
pi = np.array([[-1], [2], [5], [0], [-2], [2], [4], [5]])  # Vector columna

p=pi
print("\nVector de configuración actual p:")
print(p)

def dinamica_agentes(p, t, u):
    return u

Bbarra=np.kron(B,np.eye(2))
print("Bbarra=")
print(Bbarra)

z=Bbarra.T@p
print("z=")
print(z)
Lbarra=np.kron(L,np.eye(2))
print("Lbarra=")
print(Lbarra)




# # Parámetros del sistema
dt = 0.1
T= 100
traj = [pi.copy()]
vel=[]
# Simulación de la dinámica con Euler
for t in range(int(T / dt)):
    pdot = -Lbarra @ (p - p_ref)  # Dinámica
    vel.append(pdot.copy())
    p = p + pdot * dt  # Actualización de las posiciones con el método de Euler
    traj.append(p.copy())  # Guardar las posiciones para graficar

# Graficar la trayectoria de los agentes
traj = np.array(traj)
vel=np.array(vel)



colores = ['b', 'g', 'r', 'c']  # Colores de los 4 agentes
# Dado que p es un vector columna, hay que graficar las posiciones por separado.
plt.figure(figsize=(8, 6))
for i in range(n):
    plt.plot(traj[:, 2*i], traj[:, 2*i + 1], label=f'Agente {i+1}', color=colores[i])
    
    # Marcar el inicio con un punto del mismo color
    plt.scatter(traj[0, 2*i], traj[0, 2*i + 1], color=colores[i], marker='o')
    
    # Marcar el final con una 'X' del mismo color
    plt.scatter(traj[-1, 2*i], traj[-1, 2*i + 1], color=colores[i], marker='x')

# Añadir las líneas discontinuas negras que unen las posiciones finales de los agentes
for i in range(n - 1):
    plt.plot([traj[-1, 2 * i], traj[-1, 2 * (i + 1)]], [traj[-1, 2 * i + 1], traj[-1, 2 * (i + 1) + 1]], 'k--', linewidth=1)

# Marcar la posición de referencia
plt.plot(p_ref[::2], p_ref[1::2], 'ms',markersize=3, label='Formación de referencia')
for i in range(n - 1):
    plt.plot([p_ref[2 * i, 0], p_ref[2 * (i + 1), 0]], [p_ref[2 * i + 1, 0], p_ref[2 * (i + 1) + 1, 0]], 'm--', linewidth=1)

plt.scatter([], [], color='k', marker='o', label='Posición inicial')
plt.scatter([], [], color='k', marker='x', label='Posición final')

plt.xlabel('Posición X(-)')
plt.ylabel('Posición Y(-)')
plt.legend()
plt.title('Trayectoria de los agentes')
plt.grid(True)
plt.show()



#para pintar trayectorias y velocidades todo junto
colors=colores
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
ax1.set_xlim([0, 100])
ax1.set_ylim([-5, 5])
ax1.grid(True)
ax1.legend()

# Subplot 2 (abajo a la derecha): velocidades en Y
ax2 = fig.add_subplot(gs[1, 1])
for i in range(n):
    ax2.plot(vel[:, 2 * i + 1], color=colors[i])
ax2.set_title('Velocidades en Y(-)')
ax2.set_xlabel('Tiempo de simulación(-)')
ax2.set_ylabel('Velocidad(-)')
ax2.set_xlim([0, 100])
ax2.set_ylim([-5, 5])
ax2.grid(True)
ax2.legend()

# Subplot 3 (izquierda, ocupa ambas filas): trayectoria
ax3 = fig.add_subplot(gs[:, 0])
for i in range(n):
    ax3.plot(traj[:, 2 * i], traj[:, 2 * i + 1], color=colors[i], linewidth=1)
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
ax3.set_title('Trayectoria de los agentes, tiempo de integración = 100 (-)')
ax3.grid(True)
ax3.legend()
ax3.set_aspect('equal', adjustable='box')

# Ajustes finales
plt.suptitle('Trayectoria y convergencia de velocidades de los agentes')
plt.tight_layout()
plt.show()






def f9(t,p):
    u=-Lbarra@(p-p_ref)
    pdot=u
    return pdot








#puedo empezar en p=p0

#p=p0.copy()

print("p= (incialmente)")
print(p)

u=-Lbarra@(p-p_ref)
print("u=")
print(u)

# IV
m=2


kappa = 0.1

numnodos = 4
Z = np.array([[1, 2], [2, 3], [3, 4]])  # Aristas del grafo
mu = np.zeros((numnodos, numnodos))

np.random.seed(0)  # Para que siempre te salga igual al probarlo

# Asignamos valores aleatorios distintos a cada dirección de cada arista
#for edge in Z:
   # i, j = edge[0]-1, edge[1]-1
   # mu_ij = np.round(np.random.uniform(-0.5, 0.5), 2)
   # mu_ji = np.round(np.random.uniform(-0.5, 0.5), 2)
   # mu[i, j] = mu_ij
   # mu[j, i] = mu_ji

#resuelto a mano, para que la velocidad sea [0.3,0], como resulta ser
#le doy los valores a mano pero simplemente he resuelto el sistema de ecuaciones
#no se puede conseguir una velocidad arbitraria cualquiera (con componente y no nula)
#en este caso por las restas que se cancelan cosas y también tenemos 2 agentes
#el primero y cuarto que no tienen 2 vecinos entonces no tenemos garantizado
#ni mucho menos que tenga que poder llegarse a cualquier velocidad arbitraria

#de todas maneras, lo que importa es el caso en el que estos coeficientes son matrices 2x2

#elijo una velocidad baja para que la matriz laplaciana modificada sea relativamente 
#parecida a la original y no sea una desviación gigantesca que es lo que ocurre si coges una
#velocidad más alta al resolver el sistema


mu[0,1]=1.5
mu[3,2]=1.5
mu[1,0]=-1.5
mu[2,3]=-1.5

print("mu=")
print(mu)



omega = np.zeros((numnodos, numnodos))
for i, j in Z:
    omega[i-1, j-1] = 1
    omega[j-1, i-1] = 1  # simétrico

print("Omega original=")
print(omega)

omegamod=omega-kappa*mu
print("omega modified=")
print(omegamod)

num_aristas = len(Z)
M = np.zeros((numnodos, num_aristas))

for k, (tail,head) in enumerate(Z):  # como en el artículo
    head_idx = head - 1
    tail_idx = tail - 1
    M[tail_idx, k] = mu[tail_idx, head_idx]      # μ_{tail, head}
    M[head_idx, k] = -mu[head_idx, tail_idx]     # -μ_{head, tail}

    
    
    
print("M=")
print(M)

matrizLambda=M @ B.T  
print(" matriz Lambda=")
print(matrizLambda)

Lm=L-kappa*matrizLambda
print("Laplaciana modificada Lm=")
print(Lm)

matrizLambdabarra=np.kron(matrizLambda,np.eye(2))
print("matrizLambdabarra=")
print(matrizLambdabarra)

resultado=kappa*matrizLambdabarra @ p_ref

print("kappa·matrizLamdabarra·p*=")
print(resultado)

v_bloques = resultado.reshape((numnodos, 2))  # cada fila es una copia de v*
v_estrella = np.mean(v_bloques, axis=0)      # promedio por filas (aunque deberían ser todas iguales)
print("v* =", v_estrella)







#pdot=-Lm@p+L@p_ref

Lmbarra=np.kron(Lm,np.eye(2))
print("Laplaciana modificada ampliada con la barra, Lmbarra=")
print(Lmbarra)


pdot=-Lmbarra@p+Lbarra@p_ref

print("pdot=")
print(pdot)


#graficamos esta dinámica (la de la seccion IV A)

#empezamos en
p=np.array([[3],[5],[0],[7],[1],[2],[6],[0]])

# # Parámetros del sistema
dt = 0.1
T= 25
traj = [p.copy()]
vel=[]

# Simulación de la dinámica con Euler
for t in range(int(T / dt)):
    pdot =-Lmbarra@p+Lbarra@p_ref  # Dinámica   (20) expandida a m=2 dimensiones
    vel.append(pdot.flatten())
    p = p + pdot * dt  # Actualización de las posiciones con el método de Euler
    traj.append(p.copy())  # Guardar las posiciones para graficar

# Graficar la trayectoria de los agentes
traj = np.array(traj)
vel=np.array(vel)


colores = ['b', 'g', 'r', 'c']  # Colores de los 4 agentes
# Dado que p es un vector columna, hay que graficar las posiciones por separado.
plt.figure(figsize=(8, 6))
for i in range(n):
    plt.plot(traj[:, 2*i], traj[:, 2*i + 1], label=f'Agente {i+1}', color=colores[i])
    
    # Marcar el inicio con un punto del mismo color
    plt.scatter(traj[0, 2*i], traj[0, 2*i + 1], color=colores[i], marker='o')
    
    # Marcar el final con una 'X' del mismo color
    plt.scatter(traj[-1, 2*i], traj[-1, 2*i + 1], color=colores[i], marker='x')

# Marcar la posición de referencia
plt.plot(p_ref[::2], p_ref[1::2], 'ms',markersize=3, label='Formación de referencia')
for i in range(n - 1):
    plt.plot([p_ref[2 * i, 0], p_ref[2 * (i + 1), 0]], [p_ref[2 * i + 1, 0], p_ref[2 * (i + 1) + 1, 0]], 'm--', linewidth=1)

plt.scatter([], [], color='k', marker='o', label='Posición inicial')
plt.scatter([], [], color='k', marker='x', label='Posición final')
plt.xlabel('Posición X(-)')
plt.ylabel('Posición Y(-)')
plt.legend()
plt.title('Trayectoria de los agentes')
plt.grid(True)
plt.show()


# #intentando que se vea bien qué pasa con las velocidades
# pdots = np.array(pdots)  # shape (num_steps, 8)

# # Calcular la norma de velocidad para cada agente
# vel_norms = []
# for i in range(n):
#     v_i = pdots[:, 2*i:2*i+2]  # velocidad del agente i en cada paso (shape: num_steps x 2)
#     norm_v_i = np.linalg.norm(v_i, axis=1)  # norma de la velocidad en cada paso
#     vel_norms.append(norm_v_i)

# vel_norms = np.array(vel_norms)  # shape: (n, num_steps)


# tiempos = np.arange(vel_norms.shape[1]) * dt
# plt.figure(figsize=(9, 5))

# for i in range(n):
#     plt.plot(tiempos, vel_norms[i], label=f'Agente {i+1}', linewidth=2)

# # Línea horizontal para la norma de la velocidad deseada
# v_deseada_norm = np.linalg.norm([.3,0])
# plt.axhline(v_deseada_norm, color='k', linestyle='--', label=r'$\|\tilde{v}^*\|$')

# plt.xlabel('Tiempo (-)')
# plt.ylabel(r'$\|\dot{p}_i(t)\|$')
# plt.title('Convergencia de la norma de velocidad de cada agente')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()



#graficar las velocidades

fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

# Subfigura 1: velocidades en X
for i in range(n):
    axs[0].plot(vel[:, 2 * i], label=f'Agente {i+1}', color=colores[i])
axs[0].set_title('Velocidades en X')
axs[0].set_xlabel('Tiempo (iteraciones)')
axs[0].set_ylabel('Velocidad')
axs[0].set_ylim([-10,8])
#axs[0].set_xlim([0,200])
axs[0].grid(True)
axs[0].legend()

# Subfigura 2: velocidades en Y
for i in range(n):
    axs[1].plot(vel[:, 2 * i + 1], label=f'Agente {i+1}', color=colores[i])
axs[1].set_title('Velocidades en Y')
axs[1].set_xlabel('Tiempo (iteraciones)')
axs[1].grid(True)
axs[1].legend()

plt.suptitle('Convergencia de las velocidades de los agentes')
plt.tight_layout()
plt.show()




#graficar todo junto
colors=colores
#pintando todo lo de IV.C todo junto
# Crear la figura general
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])



# Subplot 1 (arriba a la derecha): velocidades en X
ax1 = fig.add_subplot(gs[0, 1])
for i in range(n):
    ax1.plot(vel[:, 2 * i], color=colors[i])
ax1.set_title('Velocidades en X(-)')
ax1.set_xlabel('Tiempo de integración (-)')
ax1.set_ylabel('Velocidad(-)')
ax1.set_xlim([0, 100])
ax1.set_ylim([-3, 3])
ax1.grid(True)
ax1.legend()

# Subplot 2 (abajo a la derecha): velocidades en Y
ax2 = fig.add_subplot(gs[1, 1])
for i in range(n):
    ax2.plot(vel[:, 2 * i + 1], color=colors[i])
ax2.set_title('Velocidades en Y(-)')
ax2.set_xlabel('Tiempo de integración (-)')
ax2.set_ylabel('Velocidad(-)')
ax2.set_xlim([0, 100])
ax2.set_ylim([-3, 3])
ax2.grid(True)
ax2.legend()

# Subplot 3 (izquierda, ocupa ambas filas): trayectoria
ax3 = fig.add_subplot(gs[:, 0])
for i in range(n):
    ax3.plot(traj[:, 2 * i], traj[:, 2 * i + 1], color=colors[i], linewidth=1)
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
ax3.set_title('Trayectoria de los agentes')
ax3.grid(True)
ax3.legend(loc='lower right')
ax3.set_aspect('equal', adjustable='box')

# Ajustes finales
plt.suptitle('Trayectoria y convergencia de velocidades de los agentes')
plt.tight_layout()
plt.show()





#comprobacion ec. (27)
print("primer término ec 27=",kappa**2*matrizLambdabarra@matrizLambdabarra@p_ref)

print("segundo término ec 27=",kappa**2*np.kron(M@B.T,np.eye(2))@matrizLambdabarra@p_ref)

print("tercer termino ec 27=",kappa*np.kron(M@B.T,np.eye(2))@np.kron(np.ones((4,1)),v_estrella.reshape(2,1)))


#Sección IV. C. 

#quiero calcular las matrices mu_ij

n = 4
dim = 2
kappa = 0.08

# Topología del grafo (1-2-3-4)
edges_list = [(0, 1), (1, 2), (2, 3)]
edges = set(edges_list + [(j, i) for i, j in edges_list])

# Configuración de referencia (cuadrado)
p_ref = np.array([1, 1, -1, 1, -1, -1, 1, -1]).reshape((n * dim, 1))
p_i_ref = p_ref.reshape((n, dim))

# Velocidad deseada para cada agente
v_star_i = np.array([[0.25],[ 0.0]])
v_star_apilada = np.tile(v_star_i, (n, 1))

# Inicializar contenedor de matrices mu_ij
mu_matrices = np.zeros((n, n, 2, 2))

# Construcción de mu_ij con estructura rotacional
for i in range(n):
    vecinos = [j for j in range(n) if (i, j) in edges]
    diffs = [p_i_ref[i] - p_i_ref[j] for j in vecinos]
    num_vecinos = len(vecinos)

    if num_vecinos == 1:
        # Sistema 2x2 -> solución directa
        diff = diffs[0].reshape(2, 1)
        A = np.array([[diff[0, 0], -diff[1, 0]],
                      [diff[1, 0],  diff[0, 0]]])
        b = v_star_i.reshape(2, 1) / kappa
        x = np.linalg.solve(A, b)
        alpha, beta = x.flatten()
        j = vecinos[0]
        mu_matrices[i, j] = np.array([[alpha, -beta], [beta, alpha]])
        
    elif num_vecinos > 1:
        # Sistema 2x(2*num_vecinos) sobredeterminado: fijamos beta para todos
        beta_val = 1.0 #fijo como me da la gana porque mientras se cumpla la ec.
        #14 me sirven los coeficientes que sean
        A = []
        b = v_star_i / kappa
        for d in diffs:
            A.append([d[0], -d[1]])
            A.append([d[1],  d[0]])
        A = np.array(A).reshape((2, 2 * num_vecinos))
        b = b.reshape(2, 1)

        # Separar en coeficientes de alpha (variables) y beta (fijos)
        A_alpha = A.copy()
        b_adj = b.copy()
        for k in range(num_vecinos):
            j = vecinos[k]
            d = diffs[k]
            b_adj[0] -= -beta_val * d[1]
            b_adj[1] -=  beta_val * d[0]
            A_alpha[0, 2 * k + 1] = 0
            A_alpha[1, 2 * k + 1] = 0

        # Resolver para alphas
        alphas, _, _, _ = np.linalg.lstsq(A_alpha, b_adj, rcond=None)
        for k, j in enumerate(vecinos):
            alpha = alphas[2 * k, 0]
            beta = beta_val
            mu_matrices[i, j] = np.array([[alpha, -beta], [beta, alpha]])

mu_matrices.round(3)  # Mostramos con 3 decimales para ver si salen razonables



# Verificación
verificaciones = []
for i in range(n):
    suma = np.zeros(2)
    for j in range(n):
        if mu_matrices[i, j].any():
            diff = p_i_ref[i] - p_i_ref[j]
            suma += mu_matrices[i, j] @ diff
    v_est = kappa * suma
    error = np.linalg.norm(v_est - v_star_i)
    verificaciones.append((i + 1, v_est, error))



# Verificación de la ecuación (14) para cada agente
verificaciones = []
for i in range(n):
    suma = np.zeros(2)  # Iniciar el vector de suma de la velocidad deseada
    for j in range(n):
        if (i, j) in edges:
            diff = p_i_ref[i] - p_i_ref[j]  # Diferencia entre posiciones relativas
            suma += mu_matrices[i, j] @ diff  # Sumar el efecto de mu_ij
    v_i_estimada = kappa * suma  # Velocidad estimada
    error = np.linalg.norm(v_i_estimada - v_star_i)  # Error respecto a la velocidad deseada
    verificaciones.append((i + 1, v_i_estimada, error))

mu_matrices, verificaciones

# Verificación de la ecuación (14) para cada agente
print("\nVerificación de la ecuación para cada agente:")
for i in range(n):
    suma = np.zeros(2)
    for j in range(n):
        if (i, j) in edges:
            diff = p_i_ref[i] - p_i_ref[j]
            suma += mu_matrices[i, j] @ diff
    v_i_estimada = kappa * suma
    print(f"Agente {i+1}: v_i estimada = {v_i_estimada}, v_deseada = {v_star_i}")
    print(f"Error: {np.linalg.norm(v_i_estimada - v_star_i)}\n")

# Mostrar todas las matrices mu_{ij}
print("\nMatrices μ_{ij}:")
for i in range(n):
    for j in range(n):
        print(f"\nμ_{i+1}{j+1} =")
        print(mu_matrices[i, j])



edges = np.array([[1, 2], [2, 3], [3, 4]])
num_edges = edges.shape[0]


# Creamos Msombrero (8 x 6)
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

print("Msombrero=")
print(Msombrero)

Lambdasombrero=Msombrero@Bbarra.T
print("Lambdasombrero=")
print(Lambdasombrero)

pdot=-(Lbarra-kappa*Lambdasombrero)@p+Lbarra@p_ref

print("pdot=     seccion IV. C")
print(pdot)


#seccion D

z=Bbarra.T@p

print("z=")
print(z)

z_ref=Bbarra.T@p_ref
print("z*=")
print(z_ref)

#we define the error signal
e=z-z_ref
print("e=")
print(e)

#vamos a comprobar que todos los términos de la ecuacion 36 son iguales
segundo36=-np.kron(B@D_w@B.T,np.eye(2))@(p-p_ref)+kappa*Lambdasombrero@p

tercer36=-Bbarra@e+kappa*Msombrero@z

cuarto36=-Bbarra@e+kappa*Msombrero@e+kappa*Msombrero@z_ref


v_star=v_star_i
quinto36=-Bbarra@e+kappa*Msombrero@e+np.kron(np.ones((4,1)),v_star)

print(f"primer termino36 ,pdot={pdot}")
print("segundo termino 36=",segundo36)
print("tercer termino 36=",tercer36)
print("cuarto termino 36=",cuarto36)
print("quinto termino 36=",quinto36)



#comprobación ecuacion 37 y la igualdad de arriba
zdot=Bbarra.T@pdot

edot=zdot


segundo37=-Bbarra.T@Bbarra@e+kappa*Bbarra.T@Msombrero@e+kappa*Bbarra.T@np.kron(np.ones((4,1)),v_star)

tercero37=-Bbarra.T@Bbarra@e+kappa*Bbarra.T@Msombrero@e

print(np.round(np.hstack([
    Bbarra.T @ pdot,
    zdot,
    edot,
    segundo37,
    tercero37
]), decimals=4))



#graficamos la dinámica
# # Parámetros del sistema
dt = 0.1
T= 25
p=np.array([[5],[1],[0],[6],[2],[4],[8],[9]])
traj = [p.copy()]
vel=[]
# Simulación de la dinámica con Euler
for t in range(int(T / dt)):
    pdot =-(Lbarra-kappa*Lambdasombrero)@p+Lbarra@p_ref  # Dinámica   (20) expandida a m=2 dimensiones
    vel.append(pdot.flatten().copy())

    p = p + pdot * dt  # Actualización de las posiciones con el método de Euler
    traj.append(p.copy())  # Guardar las posiciones para graficar

# Graficar la trayectoria de los agentes
traj = np.array(traj)
vel=np.array(vel)


colores = ['b', 'g', 'r', 'c']  # Colores de los 4 agentes
# Dado que p es un vector columna, hay que graficar las posiciones por separado.
plt.figure(figsize=(8, 6))
for i in range(n):
    plt.plot(traj[:, 2*i], traj[:, 2*i + 1], label=f'Agente {i+1}', color=colores[i])
    
    # Marcar el inicio con un punto del mismo color
    plt.scatter(traj[0, 2*i], traj[0, 2*i + 1], color=colores[i], marker='o')
    
    # Marcar el final con una 'X' del mismo color
    plt.scatter(traj[-1, 2*i], traj[-1, 2*i + 1], color=colores[i], marker='x')

# Marcar la posición de referencia
plt.plot(p_ref[::2], p_ref[1::2], 'ms',markersize=3, label='Formación de referencia')
for i in range(n - 1):
    plt.plot([p_ref[2 * i, 0], p_ref[2 * (i + 1), 0]], [p_ref[2 * i + 1, 0], p_ref[2 * (i + 1) + 1, 0]], 'm--', linewidth=1)

plt.scatter([], [], color='k', marker='o', label='Posición inicial')
plt.scatter([], [], color='k', marker='x', label='Posición final')
plt.xlabel('Posición X(-)')
plt.ylabel('Posición Y(-)')
plt.legend()
plt.title('Trayectoria de los agentes')
plt.grid(True)
plt.show()




fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

# Subfigura 1: velocidades en X
for i in range(n):
    axs[0].plot(vel[:, 2 * i], label=f'Agente {i+1}', color=colores[i])
axs[0].set_title('Velocidades en X')
axs[0].set_ylim([-0.5,1])
axs[0].set_xlabel('Tiempo (iteraciones)')
axs[0].set_ylabel('Velocidad')
axs[0].grid(True)
axs[0].legend()

# Subfigura 2: velocidades en Y
for i in range(n):
    axs[1].plot(vel[:, 2 * i + 1], label=f'Agente {i+1}', color=colores[i])
axs[1].set_title('Velocidades en Y')
axs[0].set_ylim([-0.5,1])
axs[1].set_xlabel('Tiempo (iteraciones)')
axs[1].grid(True)
axs[1].legend()

plt.suptitle('Convergencia de las velocidades de los agentes')
plt.tight_layout()
plt.show()



#graficar todo lo de IV.C junto

colors=colores
#pintando todo lo de IV.C todo junto
# Crear la figura general
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])



# Subplot 1 (arriba a la derecha): velocidades en X
ax1 = fig.add_subplot(gs[0, 1])
for i in range(n):
    ax1.plot(vel[:, 2 * i], color=colors[i])
ax1.set_title('Velocidades en X(-)')
ax1.set_xlabel('Tiempo de integración (-)')
ax1.set_ylabel('Velocidad(-)')
ax1.set_xlim([0, 100])
ax1.set_ylim([-3, 3])
ax1.grid(True)
ax1.legend()

# Subplot 2 (abajo a la derecha): velocidades en Y
ax2 = fig.add_subplot(gs[1, 1])
for i in range(n):
    ax2.plot(vel[:, 2 * i + 1], color=colors[i])
ax2.set_title('Velocidades en Y(-)')
ax2.set_xlabel('Tiempo de integración (-)')
ax2.set_ylabel('Velocidad(-)')
ax2.set_xlim([0, 100])
ax2.set_ylim([-3, 3])
ax2.grid(True)
ax2.legend()

# Subplot 3 (izquierda, ocupa ambas filas): trayectoria
ax3 = fig.add_subplot(gs[:, 0])
for i in range(n):
    ax3.plot(traj[:, 2 * i], traj[:, 2 * i + 1], color=colors[i], linewidth=1)
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
ax3.set_title('Trayectoria de los agentes')
ax3.grid(True)
ax3.legend()
ax3.set_aspect('equal', adjustable='box')

# Ajustes finales
plt.suptitle('Trayectoria y convergencia de velocidades de los agentes')
plt.tight_layout()
plt.show()








# =============================================================================
# SECCIÓN V - CASO SIMPLE: DOS AGENTES EN 2D, UN ERROR DE ESCALADO EN LA MEDICIÓN
# =============================================================================



# Parámetros
a = 1.2 # factor de escalado incorrecto del agente 2
dt = 0.01
T = 75
steps = int(T / dt)

b=(a-1)/2
c=(a-1)/2

# Posiciones iniciales (vectores columna)
p1_0 = np.array([[1.0], [3.0]])  # posición inicial del agente 1
p2_0 = np.array([[10.0], [7.0]])  # posición inicial del agente 2

# Objetivo relativo deseado z_12^* = p1^* - p2^*
z12_ref = np.array([[1.5], [4.0]])

# Para almacenar trayectorias
traj_p1 = [p1_0.copy()]
traj_p2 = [p2_0.copy()]

# Iteración para cada paso de tiempo
for _ in range(steps):
    # Z_12 = p1 - p2
    z12 = traj_p1[-1] - traj_p2[-1]
    
    # Dinámica según ecuación (41) del artículo con escala incorrecta en agente 2
    dp1 = -((b + 1) * z12 - z12_ref) + b * z12
    dp2 = ((a - c) * z12 - z12_ref) + c * z12

    # Euler
    p1_new = traj_p1[-1] + dt * dp1
    p2_new = traj_p2[-1] + dt * dp2

    traj_p1.append(p1_new.copy())
    traj_p2.append(p2_new.copy())

# Conversión para graficar
traj_p1 = np.hstack(traj_p1)
traj_p2 = np.hstack(traj_p2)

# Velocidades finales
v1_final = dp1.flatten()
v2_final = dp2.flatten()

# Distorsión final
z12_final = traj_p1[:, -1] - traj_p2[:, -1]

# Vectores de posición relativa en el tiempo
z12_traj = traj_p1 - traj_p2



# Conversión de listas a arrays para análisis
traj_p1 = np.array(traj_p1).reshape(2, -1)  # shape: (2, steps + 1)
traj_p2 = np.array(traj_p2).reshape(2, -1)  # shape: (2, steps + 1)

# Velocidades finales
v1_final = (traj_p1[:, -1] - traj_p1[:, -2]) / dt
v2_final = (traj_p2[:, -1] - traj_p2[:, -2]) / dt
print("Velocidad final agente 1:", v1_final)
print("Velocidad final agente 2:", v2_final)

#la velocidad residual es menor cuanto más cercano a 1 es a



z12_traj = traj_p1 - traj_p2  # trayectoria relativa



# Distorsión final (último valor de z12)
z12_final = z12_traj[:, -1]
print("Distorsión final z_12:", z12_final)
print("z_12 deseado:", z12_ref.flatten())

# Comparación con la velocidad teórica esperada
v_teorica = (a - 1) / (a + 1) * z12_ref.flatten()  #comprobación ec. (34) de mi tfg
print("Velocidad teórica esperada:", v_teorica)

# ========= Gráficas adicionales ==========

# Gráfico 1: trayectoria de los agentes
plt.figure(figsize=(6, 6))
plt.plot(traj_p1[0], traj_p1[1], label='Agente 1')
plt.plot(traj_p2[0], traj_p2[1], label='Agente 2')

# Marcar los puntos iniciales
plt.scatter(p1_0[0], p1_0[1], c='blue', marker='o', label='Inicio Agente 1')
plt.scatter(p2_0[0], p2_0[1], c='orange', marker='o', label='Inicio Agente 2')

# Marcar los puntos finales con una "X"
plt.scatter(traj_p1[0, -1], traj_p1[1, -1], c='blue', marker='x', label='Final Agente 1')
plt.scatter(traj_p2[0, -1], traj_p2[1, -1], c='orange', marker='x', label='Final Agente 2')

# Títulos y etiquetas
plt.title("Trayectorias de los agentes")
plt.xlabel("x(-)")
plt.ylabel("y(-)")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()






#pasamos al caso general de la seccion V
p = np.array([[2], [0], [3], [8], [7], [1], [4], [7]])

n=4 #agentes
m=2 #dimensiones


a=[1.1,0.75,0.93,1.34]

#necesitamos hacer unas cuantas cosas para calcular D_R
theta=[0.06,0.24,-0.16,-0.09]

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
    print("R_",i+1,".T=",R_i.T)
    listaR_RiT.append(R_i.T)

print("listaR_RiT",listaR_RiT)    
R=np.hstack(listaR_RiT).T


def bloque_diagonal(lista_de_matrices):
    n = len(lista_de_matrices)
    m = lista_de_matrices[0].shape[0]
    
    return np.block([
        [lista_de_matrices[i] if i == j else np.zeros((m, m)) for j in range(n)]
        for i in range(n)
    ])
    
D_R=bloque_diagonal(listaR)

print("R=")
print(R)

print("D_R=")
print(np.array2string(D_R,precision=4))

#lo puedo hacer así, pero al final es más fácil teniendo 
#a como una lista, hacerlo como se hace después del comentario

# a=[np.array([[1.1]]),np.array([[.75]]),np.array([[.93]]),np.array([[1.34]])]
# Da=bloque_diagonal(a)


Da=np.diag(a)
print("Da=")
print(Da)

Dabarra=np.kron(Da,np.eye(2))

print("Da_barra=")
print(Dabarra)

Dx=Dabarra@D_R
print("Dx=")
print(np.array2string(Dx,precision=4))

#pdot=-Dx@Lbarra@p+Lbarra@p_ref

# # Parámetros del sistema
dt = 0.1
T= 15
traj = [p.copy()]
pdots=[]
# Simulación de la dinámica con Euler
for t in range(int(T / dt)):
    pdot = -Dx@Lbarra@p+Lbarra@p_ref # Dinámica
    pdots.append(pdot.copy())
    p = p + pdot * dt  # Actualización de las posiciones con el método de Euler
    traj.append(p.copy())  # Guardar las posiciones para graficar

# Graficar la trayectoria de los agentes
traj = np.array(traj)
pdots=np.array(pdots)




colores = ['b', 'g', 'r', 'c']  # Colores de los 4 agentes
# Dado que p es un vector columna, hay que graficar las posiciones por separado.
plt.figure(figsize=(8, 6))
for i in range(n):
    plt.plot(traj[:, 2*i], traj[:, 2*i + 1], label=f'Agente {i+1}', color=colores[i])
    
    # Marcar el inicio con un punto del mismo color
    plt.scatter(traj[0, 2*i], traj[0, 2*i + 1], color=colores[i], marker='o')
    
    # Marcar el final con una 'X' del mismo color
    plt.scatter(traj[-1, 2*i], traj[-1, 2*i + 1], color=colores[i], marker='x')

# Marcar la posición de referencia

# Marcar la posición de referencia
plt.plot(p_ref[::2], p_ref[1::2], 'ms',markersize=3, label='Formación de referencia')
for i in range(n - 1):
    plt.plot([p_ref[2 * i, 0], p_ref[2 * (i + 1), 0]], [p_ref[2 * i + 1, 0], p_ref[2 * (i + 1) + 1, 0]], 'm--', linewidth=1)

plt.scatter([], [], color='k', marker='o', label='Posición inicial')
plt.scatter([], [], color='k', marker='x', label='Posición final')

plt.xlabel('Posición X(-)')
plt.ylabel('Posición Y(-)')
plt.legend()
plt.title('Trayectoria de los agentes')
plt.grid(True)
plt.show()


#pintar trayectorias y evolución de velocidades todo junto
colors=colores

# Crear la figura general
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])



# Subplot 1 (arriba a la derecha): velocidades en X
ax1 = fig.add_subplot(gs[0, 1])
for i in range(n):
    ax1.plot(pdots[:, 2 * i], color=colors[i])
ax1.set_title('Velocidades en X(-)')
ax1.set_xlabel('Tiempo de integración (-)')
ax1.set_ylabel('Velocidad(-)')
ax1.set_xlim([0, 100])
ax1.set_ylim([-3, 3])
ax1.grid(True)
ax1.legend()

# Subplot 2 (abajo a la derecha): velocidades en Y
ax2 = fig.add_subplot(gs[1, 1])
for i in range(n):
    ax2.plot(pdots[:, 2 * i + 1], color=colors[i])
ax2.set_title('Velocidades en Y(-)')
ax2.set_xlabel('Tiempo de integración (-)')
ax2.set_ylabel('Velocidad(-)')
ax2.set_xlim([0, 100])
ax2.set_ylim([-3, 3])
ax2.grid(True)
ax2.legend()

# Subplot 3 (izquierda, ocupa ambas filas): trayectoria
ax3 = fig.add_subplot(gs[:, 0])
for i in range(n):
    ax3.plot(traj[:, 2 * i], traj[:, 2 * i + 1], color=colors[i], linewidth=1)
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
ax3.set_title('Trayectoria de los agentes')
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

print("Mcuenca=")
print(Mcuenca)

#ecuacion 46 (despejo del primer término, pero no sé si la matriz va a ser invertible)

ptilde_ref=np.linalg.solve((Dx@np.kron(B@D_w,np.eye(2))+Mcuenca)@Bbarra.T,Lbarra@p_ref)

print("ptilde_ref=")
print(ptilde_ref)




#voy a calcular v˜* 

vtilde_ref_full = Mcuenca @ Bbarra.T@ptilde_ref  # ∈ ℝ^{mn}
vtilde_ref_blocks = vtilde_ref_full.reshape((n, m))  # n filas, cada una es vtilde_i

# Calcular promedio (asume que todos tienen el mismo vtilde*)
vtilde_ref = np.mean(vtilde_ref_blocks, axis=0).reshape((m, 1)) #son todos iguales

print("vtilde_ref=")
print(vtilde_ref)

np.savez('datos_caso1.npz', traj=traj, pdots=pdots, ptilde_ref=ptilde_ref, Bbarra=Bbarra, vtilde_ref=vtilde_ref)


#al dibujarlo ptilde_ref, vemos que el ptilde_ref que se alcanza es bastante parecido a
#p_ref, más parecido cuando menores sean los errores introducidos



n = int(p_ref.shape[0] // 2)

p_ref_2D = p_ref.reshape((n, 2))
ptilde_ref_2D = ptilde_ref.reshape((n, 2))

#print(ptilde_ref_2D)

# Crear el gráfico
plt.figure(figsize=(6, 6))
plt.plot(p_ref_2D[:, 0], p_ref_2D[:, 1], 'bo-', label='Formación original de referencia (p_ref)')
plt.plot(ptilde_ref_2D[:, 0], ptilde_ref_2D[:, 1], 'ro--', label='Formación alcanzada distorsionada ')

# Añadir etiquetas a cada agente
for i in range(n):
    plt.text(p_ref_2D[i, 0] + 0.05, p_ref_2D[i, 1] + 0.05, f'{i+1}', color='blue')
    plt.text(ptilde_ref_2D[i, 0] + 0.05, ptilde_ref_2D[i, 1] - 0.1, f'{i+1}', color='red')

plt.title('Comparación de formaciones: p_ref vs. ptilde_ref')
plt.xlabel('Posición X(-)')
plt.ylabel('Posición Y(-)')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()


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

errores = errores[1:]
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



# # Graficar la evolución del error
# plt.figure(figsize=(8, 5))
# plt.plot(np.arange(len(errores)) * dt, errores, label=r'$\|z(t) - \tilde{z}^*\|$')
# plt.xlabel('Tiempo (-)')

# plt.title(r'Convergencia de $z(t)$ hacia $\tilde{z}^*$')
# plt.grid(True)
# plt.legend(prop={'size': 14})  
# plt.tight_layout()
# plt.show()