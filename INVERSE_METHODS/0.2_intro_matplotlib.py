#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np

# Equally spaced points
# From 0 to 10 with a spacing of 0.2
x = np.arange(0,10,0.2)
y = x**2 + 1

plt.plot(x,y)
plt.title('y=x**2+1')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# Graphics with points
plt.plot(x,y,'.')
plt.show()


# Two graphics in one panel
# Red crosses and blue line
plt.plot(x,y,'xr')
plt.plot(x,y,'-b')
plt.show()


# Two graphics in one panel
# Black crosses and green dotted line
plt.plot(x,y,'xk', x,y,':g')
plt.show()


# Changing marker size 
plt.plot(x,y,'xk', markersize=10)
plt.show()


# Changing line width
plt.plot(x,y,'r-', linewidth=5)
plt.show()


# Changing axis range
plt.plot(x,y,'k-')
plt.xlim([2,8])
plt.ylim([20,60])
plt.show()


# Adding labels
plt.plot(x,y,'k-')
plt.annotate('Hola!', xy=(4,16), xytext=(1, 80) )
plt.annotate('y=x**2+1', xy=(4, 16), xytext=(2, 40), arrowprops=dict(facecolor='black') )
plt.show()


# Logarithmic scale
plt.plot(x,y,'k-')
plt.yscale('log')
plt.xscale('log')
plt.show()


# Polar diagrams
ang = np.arange(0,2*np.pi,0.1)
r = np.cos(2*ang) + 1.5
ax = plt.subplot(111, projection='polar')
ax.plot(ang,r,'kx')
plt.show()


# Multiple plots

x1 = np.arange(0.0, 5.0, 0.1)
y1 = np.cos(2*np.pi*x1)

plt.figure(1)

plt.subplot(3,1,1)
plt.plot(x1, y1, 'bo')

plt.subplot(3,1,2)
plt.plot(x1, y1, 'k')

plt.subplot(3,1,3)
plt.plot(x1, y1, 'r--')

plt.show()


# Multiple graphics with an arbitrary function

def f(t):
    return np.cos(2*np.pi*t) * np.exp(-((t-2.5)/2)**2)

x1 = np.arange(0.0, 5.0, 0.1)

plt.figure()

plt.subplot(3,1,1)
plt.plot(x1, f(x1), 'bo')

plt.subplot(3,1,2)
plt.plot(x1, f(x1+1), 'k')
plt.xlim([2,4])

plt.subplot(3,1,3)
plt.plot(x1, f(x1*2), 'r--')

plt.show()

