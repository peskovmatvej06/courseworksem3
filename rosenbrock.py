import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


SEED = 42
np.random.seed(SEED)

def exp_rosenbrock():
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Запуск эксперимента: Функция Розенброка (3D)...")
    
    
    def rosenbrock_func(x, y):
        return (1 - x)**2 + 100 * (y - x**2)**2

    
    def ros_grad(v):
        x, y = v[0], v[1]
        dx = -400 * x * (y - x**2) - 2 * (1 - x)
        dy = 200 * (y - x**2)
        return np.array([dx, dy])

    
    def run(x0, lr, steps):
        path = [x0]
        x = x0.copy()
        for _ in range(steps):
            x = x - lr * ros_grad(x)
            path.append(x)
            
            if np.max(np.abs(x)) > 5: break
        return np.array(path)

    
    x0 = np.array([-1.5, 2.0]) 
    
   
    path_ode = run(x0, 0.00002, 10000)
    z_ode = [rosenbrock_func(p[0], p[1]) for p in path_ode]
    
    
    path_gd = run(x0, 0.002, 500)
    z_gd = [rosenbrock_func(p[0], p[1]) for p in path_gd]
    
    
    path_uns = run(x0, 0.0042, 50)
    z_uns = [rosenbrock_func(p[0], p[1]) for p in path_uns]

    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    
    x_range = np.linspace(-2, 2.5, 100)
    y_range = np.linspace(-1, 3.5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = rosenbrock_func(X, Y)
    
    
    Z_plot = np.clip(Z, 0, 150)

    
    ax.plot_surface(X, Y, Z_plot, cmap='viridis', alpha=0.4, edgecolor='none')
    
    
    ax.contour(X, Y, Z, zdir='z', offset=0, levels=30, cmap='viridis', alpha=0.5)

   
    ax.plot(path_ode[:,0], path_ode[:,1], z_ode, 'k-', lw=2.5, label='ODE Flow')
    ax.plot(path_gd[:,0], path_gd[:,1], z_gd, 'b-', lw=1.5, label='Stable GD')
    ax.plot(path_uns[:,0], path_uns[:,1], z_uns, 'r--', lw=1.5, label='Unstable GD')

   
    ax.scatter(1, 1, 0, color='red', s=100, marker='*', label='Global Min', zorder=10)

    
    ax.set_title("Rosenbrock Valley: Discrete vs Continuous")
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$y$', fontsize=12)
    ax.set_zlabel('$f(x, y)$', fontsize=12)
    
    ax.set_zlim(0, 100)           
    ax.view_init(elev=45, azim=-130) 
    ax.legend()
    
    
    output_path = os.path.join(script_dir, "rosenbrock.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Картинка сохранена: {output_path}")
    plt.close()

if __name__ == "__main__":
    exp_rosenbrock()