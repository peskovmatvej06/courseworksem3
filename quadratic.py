import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


SEED = 42
np.random.seed(SEED)

def exp_quadratic():
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Запуск эксперимента: Квадратичная функция (Граница 2/L)...")
    
   
    A = np.array([[1.0, 0.0], 
                  [0.0, 10.0]])
    
    def func(x, y):
        return 0.5 * (1.0 * x**2 + 10.0 * y**2)

    def grad(v):
        
        return A @ v

    
    def run_gd(x0, lr, steps):
        path = [x0]
        x = x0.copy()
        for _ in range(steps):
            g = grad(x)
            x = x - lr * g
            path.append(x)
           
            if np.max(np.abs(x)) > 10: break
        return np.array(path)

   
    x0 = np.array([4.0, 2.0])

    
    
   
    path_ode = run_gd(x0, lr=0.005, steps=500)
    z_ode = [func(p[0], p[1]) for p in path_ode]

    
    path_stable = run_gd(x0, lr=0.19, steps=30)
    z_stable = [func(p[0], p[1]) for p in path_stable]

    
    path_uns = run_gd(x0, lr=0.205, steps=10)
    z_uns = [func(p[0], p[1]) for p in path_uns]

   
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

   
    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = func(X, Y)

    
    ax.plot_surface(X, Y, Z, cmap='Blues', alpha=0.2, edgecolor='none')
    
    ax.contour(X, Y, Z, zdir='z', offset=0, levels=15, cmap='Blues', alpha=0.5)

   
    ax.plot(path_ode[:,0], path_ode[:,1], z_ode, 'k-', lw=2.5, label=r'ODE Flow ($\eta \to 0$)')
    ax.plot(path_stable[:,0], path_stable[:,1], z_stable, 'b.-', lw=1.5, label=r'Stable ($\eta < 2/L$)')
    ax.plot(path_uns[:,0], path_uns[:,1], z_uns, 'r.--', lw=1.5, label=r'Unstable ($\eta > 2/L$)')

   
    ax.set_title("Stability Analysis: Quadratic Case ($L=10$)")
    ax.set_xlabel('$x_1$ (Small curvature)')
    ax.set_ylabel('$x_2$ (Large curvature)')
    ax.set_zlabel('$f(x)$')
    ax.set_zlim(0, 50) 
    
    ax.view_init(elev=30, azim=-60)
    ax.legend()

   
    output_path = os.path.join(script_dir, "quadratic.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Картинка сохранена: {output_path}")
    plt.close()

if __name__ == "__main__":
    exp_quadratic()