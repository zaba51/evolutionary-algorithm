import numpy as np
import matplotlib.pyplot as plt


def plot_function_3d(func, file_path):
    n = 30
    sampled = np.linspace(-10, 10, n)
    x, y = np.meshgrid(sampled, sampled)
    z = np.zeros((len(sampled), len(sampled)))

    for i in range(len(sampled)):
        for j in range(len(sampled)):
            z[i, j] = func(np.array([x[i, j], y[i, j]]))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                           linewidth=0.4, antialiased=True)
    ax.view_init(30, 200)
    plt.savefig(file_path)
    plt.close()


def plot_population_3d(population, values, title, path_to_file):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = [ind[0] for ind in population]
    ys = [ind[1] for ind in population]
    zs = values
    ax.scatter(xs, ys, zs, color='green')
    best_index = np.argmin(zs)
    best_x = xs[best_index]
    best_y = ys[best_index]
    best_z = zs[best_index]
    ax.scatter([best_x], [best_y], [best_z], color='red', s=50)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    ax.view_init(elev=15, azim=45)
    plt.savefig(path_to_file)
    plt.close()


def plot_best(best_values_per_generation, title, path_to_file):
    generations = list(range(0, len(best_values_per_generation)))
    min_line = np.zeros(len(generations))
    plt.figure()
    plt.plot(generations, best_values_per_generation, color='blue')
    plt.plot(generations, min_line, color='red')
    plt.xlabel('Generacja')
    plt.ylabel('Najlepsza wartość')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path_to_file)
    plt.close()
