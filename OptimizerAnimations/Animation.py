#http://tiao.io/notes/visualizing-and-animating-optimization-algorithms-with-matplotlib/
import matplotlib.pyplot as plt
import autograd.numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import animation
from IPython.display import HTML

from autograd import elementwise_grad, value_and_grad, grad
from scipy.optimize import minimize
from collections import defaultdict
from itertools import zip_longest
from functools import partial

f  = lambda x, y: (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
xmin, xmax, xstep = -4.5, 4.5, .2
ymin, ymax, ystep = -4.5, 4.5, .2

x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
z = f(x, y)


minima = np.array([3., .5])
minima_ = minima.reshape(-1, 1)

x0 = np.array([1., 1.])

def make_minimize_cb(path=[]):
    
    def minimize_cb(xk):
        # note that we make a deep copy of xk
        path.append(np.copy(xk))

    return minimize_cb


class TrajectoryAnimation(animation.FuncAnimation):
    
    def __init__(self, *paths, labels=[], fig=None, ax=None, frames=None, interval=60, repeat_delay=5, blit=True, **kwargs):

        if fig is None:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.get_figure()
        else:
            if ax is None:
                ax = fig.gca()

        self.fig = fig
        self.ax = ax
        
        self.paths = paths

        if frames is None:
            frames = max(path.shape[1] for path in paths)
  
        self.lines = [ax.plot([], [], label=label, lw=2)[0] 
                      for _, label in zip_longest(paths, labels)]
        self.points = [ax.plot([], [], 'o', color=line.get_color())[0] 
                       for line in self.lines]

        super(TrajectoryAnimation, self).__init__(fig, self.animate, init_func=self.init_anim,
                                                  frames=frames, interval=interval, blit=blit,
                                                  repeat_delay=repeat_delay, **kwargs)

    def init_anim(self):
        for line, point in zip(self.lines, self.points):
            line.set_data([], [])
            point.set_data([], [])

        x = self.lines + self.points
        return x

    def animate(self, i):
        for line, point, path in zip(self.lines, self.points, self.paths):
            line.set_data(*path[::,:i])
            point.set_data(*path[::,i-1:i])

        x = self.lines + self.points
        return x


methods = [
    "SGD",
    "Momentum"
]

def stochasticGradientDescent(function, x0, y0, learning_rate, num_steps):

    allX = [x0]
    allY = [y0]

    x = x0
    y = y0

    for _ in range(num_steps):
        
        dz_dx = grad(function, argnum=0)(x, y)
        dz_dy = grad(function, argnum=1)(x, y)

        x = x - dz_dx * learning_rate
        y = y - dz_dy * learning_rate

        allX.append(x)
        allY.append(y)

    return np.array([allX, allY])

def momentumUpdate(function, x0, y0, learning_rate, num_steps, momentum = 0.9):


    allX = [x0]
    allY = [y0]

    x = x0
    y = y0
    
    x_v = 0
    y_v = 0

    for _ in range(num_steps):
        
        dz_dx = grad(function, argnum=0)(x, y)
        dz_dy = grad(function, argnum=1)(x, y)
        
        x_v = (momentum * x_v) - (dz_dx) * learning_rate
        y_v = (momentum * y_v) - (dz_dy) * learning_rate

        x = x + x_v
        y = y + y_v

        allX.append(x)
        allY.append(y)

    return np.array([allX, allY])


sgdPath = stochasticGradientDescent(f, x0[0], x0[1], 0.005, 100)
momentumPath =  momentumUpdate(f, x0[0], x0[1], 0.005, 100)

paths = [sgdPath, momentumPath]

fig, ax = plt.subplots(figsize=(10, 6))

ax.contour(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
ax.plot(*minima_, 'r*', markersize=10)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))

anim = TrajectoryAnimation(*paths, labels=methods, ax=ax)

ax.legend(loc='upper left')

html = "<html><body>" + anim.to_html5_video() + "</body></html>"
file = open('animation.html', 'w')
file.write(html)
file.close()