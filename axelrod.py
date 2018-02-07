import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import copy


# klasa definiująca cechy pojedynczego automatu (agenta)
class Agent:
    def __init__(self, color, ingroup_coop, outgroup_coop):
        self.color = color
        self.inner = ingroup_coop
        self.outer = outgroup_coop
        self.ptr = 0.12

    def give(self):
        self.ptr -= 0.01

    def receive(self):
        self.ptr += 0.03

    def reset(self):
        self.ptr = 0.12


# klasa modelu przestrzeni symulacyjnej (tablica dwuwymiarowa)
class World:
    def __init__(self, size):
        self.size = size
        self.grid = [[None for j in range(size)] for i in range(size)]

    def visualization(self):  # metoda zwracająca strukturę
        g = self.grid
        visual_data = {'x': [], 'y': [], 'c': [], 'edgecolors': []}
        for x in range(self.size):
            for y in range(self.size):
                if g[x][y] is not None:
                    visual_data['x'].append(x)
                    visual_data['y'].append(y)

                    if g[x][y].inner and g[x][y].outer:  # samarytanizm
                        visual_data['c'].append(g[x][y].color)
                        visual_data['edgecolors'].append('w')

                    elif g[x][y].inner and not g[x][y].outer:  # etnocentryzm
                        visual_data['c'].append(g[x][y].color)
                        visual_data['edgecolors'].append(g[x][y].color)

                    elif not g[x][y].inner and g[x][y].outer:  # zdrada
                        visual_data['c'].append('w')
                        visual_data['edgecolors'].append(g[x][y].color)

                    elif not g[x][y].inner and not g[x][y].outer:  # egozim
                        visual_data['c'].append('k')
                        visual_data['edgecolors'].append(g[x][y].color)
        return visual_data


# klasa instancji obiektu przeprowadzającego proces symulacji
class Simulation:
    def __init__(self, size):
        self.size = size
        self.world = World(size)
        self.colors = ['c', 'm', 'y', 'r']
        self.mutation_rate = 0.005
        self.death_probability = 0.10

    def __randomize_features(self):  # metoda generująca losowe wartości dla poszczególnych cech nowego automatu (agenta)
        return np.random.choice(self.colors), np.random.randint(0, 2), np.random.randint(0, 2)

    @staticmethod
    def __prisoners_dilemma(a, b):  # logika intekacji dla relacji wewnątrzgrupowej i międzygrupowej
        if a.color == b.color:
            if a.inner and b.inner:
                a.give()
                a.receive()
                b.give()
                b.receive()
            elif a.inner and not b.inner:
                a.give()
                b.receive()
            elif not a.inner and b.inner:
                b.give()
                a.receive()
            elif not a.inner and not b.inner:
                pass

        if a.color != b.color:
            if a.outer and b.outer:
                a.give()
                b.receive()
                b.give()
                a.receive()
            elif a.outer and not b.outer:
                a.give()
                b.receive()
            elif not a.outer and b.outer:
                b.give()
                a.receive()
            elif not a.outer and not b.outer:
                pass

    def settlement(self):  # metoda zasiedlająca losową wolną komórkę w przestrzeni symulacyjnej
        g = self.world.grid
        for i in np.random.permutation(self.size):
            for j in np.random.permutation(self.size):
                if g[i][j] is None:
                    g[i][j] = Agent(*self.__randomize_features())
                    return

    def interaction(self):  # metoda parująca agentów pozostających w interkacji
        g = self.world.grid
        for i in range(self.size):
            for j in range(self.size):
                a = g[i][j]
                b = g[i][(j + 1) % self.size]
                if a is not None and b is not None:
                    self.__prisoners_dilemma(a, b)
        for j in range(self.size):
            for i in range(self.size):
                a = g[i][j]
                b = g[(i + 1) % self.size][j]
                if a is not None and b is not None:
                    self.__prisoners_dilemma(a, b)

    def reproduction(self):  # metoda reprodukująca agentów w oparciu o współczynnik PTR
        g = self.world.grid
        for i in np.random.permutation(self.size):
            for j in np.random.permutation(self.size):
                if g[i][j] is not None:
                    if np.random.rand() < g[i][j].ptr:
                        up = ((i - 1) % self.size, j)
                        down = ((i + 1) % self.size, j)
                        left = (i, (j - 1) % self.size)
                        right = (i, (j + 1) % self.size)
                        options = [up, down, left, right]
                        order = np.random.permutation(4)
                        for k in order:
                            if g[options[k][0]][options[k][1]] is None:
                                g[options[k][0]][options[k][1]] = copy.deepcopy(g[i][j])
                                # mutacja cech potomstwa
                                if np.random.rand() < self.mutation_rate:
                                    g[options[k][0]][options[k][1]].color = np.random.choice(self.colors)
                                if np.random.rand() < self.mutation_rate:
                                    g[options[k][0]][options[k][1]].inner = np.random.randint(2)
                                if np.random.rand() < self.mutation_rate:
                                    g[options[k][0]][options[k][1]].outer = np.random.randint(2)
                                break
        # przywracanie współczynnika potencjału reprodukcyjnego (PTR) do stanu początkowego
        for i in range(self.size):
            for j in range(self.size):
                if g[i][j] is not None:
                    g[i][j].reset()

    # metoda losowo uśmiercająca wybranych agentów w oparciu o przyjęty współczynnik prawdopodobieństwa
    def death(self):
        g = self.world.grid
        for i in range(self.size):
            for j in range(self.size):
                if np.random.rand() < self.death_probability:
                    g[i][j] = None

    def statistics(self):
        total = 0
        ethnocentric = 0
        samaritan = 0
        traitor = 0
        selfish = 0
        g = self.world.grid
        for i in range(self.size):
            for j in range(self.size):
                if g[i][j] is not None:
                    total += 1
                    if g[i][j].inner and not g[i][j].outer:
                        ethnocentric += 1
                    elif g[i][j].inner and g[i][j].outer:
                        samaritan += 1
                    elif not g[i][j].inner and g[i][j].outer:
                        traitor += 1
                    elif not g[i][j].inner and not g[i][j].outer:
                        selfish += 1
        return [ethnocentric, samaritan, traitor, selfish]


###################################################################################################
plt.rcParams['animation.ffmpeg_path'] = 'D:\\ffmpeg-20180202-caaa40d-win64-static\\bin\\ffmpeg.exe'
size = 50
simulation = Simulation(size)
fig, ax = plt.subplots()
fig.set_size_inches(6.5, 6.5, True)
scat = ax.scatter([], [])


def perform_simulation(iterations=1000):
    simulation = Simulation(50)
    agents_stat = [[], [], [], []]

    for i in range(iterations):
        simulation.settlement()
        simulation.interaction()
        simulation.reproduction()
        simulation.death()

        result = simulation.statistics()
        agents_stat[0].append(result[0])
        agents_stat[1].append(result[1])
        agents_stat[2].append(result[2])
        agents_stat[3].append(result[3])

    data = {'Etnocentryzm': pd.Series(agents_stat[0]),
            'Samarytanizm': pd.Series(agents_stat[1]),
            'Zdrada': pd.Series(agents_stat[2]),
            'Egoizm': pd.Series(agents_stat[3])}

    df = pd.DataFrame(data)

    df.plot()
    plt.title('Model Axelroda-Hammonda')
    plt.ylabel('Liczba agentów')
    plt.xlabel('Iteracje')
    plt.tight_layout()
    plt.savefig('axelrod-hammond.png')
    plt.close()


def init():
    ax.set(xlim=(-4, size + 4), ylim=(-4, size + 4))
    ax.set_facecolor('0.85')
    return scat,


def animate(frame):
    simulation.settlement()
    simulation.interaction()
    simulation.reproduction()
    simulation.death()

    ax.clear()
    ax.set(xlim=(-4, size + 4), ylim=(-4, size + 4))
    data = simulation.world.visualization()
    ax.scatter(data['x'], data['y'], c=data['c'], edgecolors=data['edgecolors'], marker='s')
    return scat,


def main():

    # ani = animation.FuncAnimation(fig, animate, init_func=init, frames=1000, blit=True)
    # ani.save('ethno.mp4', writer=animation.FFMpegFileWriter(), dpi=150)
    # plt.show()

    perform_simulation(1000)


if __name__ == "__main__":
    main()
