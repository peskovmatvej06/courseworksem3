import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


SEED = 42
np.random.seed(SEED)

def exp_sgd_cloud():
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Запуск эксперимента: SGD Cloud (3D)...")
    
    
    def loss_func(x, y):
        return 0.5 * (x**2 + 4 * y**2)

   
    def true_grad(x): 
        return np.array([x[0], 4 * x[1]])
    
    
    def noisy_grad(x):
        g = true_grad(x)
  
        noise = np.random.normal(0, 1.5, size=x.shape) 
        return g + noise

   
    x = np.array([4.0, 3.0])
    lr = 0.05
    steps = 3000
    path = []
    
    
    for _ in range(steps):
        
        z_val = loss_func(x[0], x[1])
        path.append([x[0], x[1], z_val])
        
        
        x = x - lr * noisy_grad(x)
        
    path = np.array(path)

    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

   
    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = loss_func(X, Y)
    
    
    ax.plot_surface(X, Y, Z, cmap='Blues', alpha=0.3, edgecolor='none')
    
    
    ax.contour(X, Y, Z, zdir='z', offset=0, levels=15, cmap='Blues', alpha=0.5)

    
    ax.plot(path[:50, 0], path[:50, 1], path[:50, 2], 
            'b-', lw=2, label='Initial Descent', zorder=10)

    
    cloud = path[100:]
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], 
               c='r', s=5, alpha=0.2, label='Stationary Cloud', zorder=5)

    
    ax.set_title("SGD Dynamics: Descent & Diffusion")
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$y$', fontsize=12)
    ax.set_zlabel('$f(x, y)$', fontsize=12)
    ax.set_zlim(0, 40) 
   
    ax.view_init(elev=35, azim=-60)
    
    ax.legend()

    
    output_path = os.path.join(script_dir, "sgd_cloud.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Картинка сохранена: {output_path}")
    plt.close()

if __name__ == "__main__":
    exp_sgd_cloud()