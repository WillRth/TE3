#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:18:02 2023

@author: etu_28604517
"""

import numpy as np
import time
from scipy import linalg
import colorsys
from PIL import Image
import matplotlib.pyplot as plt
from numpy import complex
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from numba import cuda, jit
from timeit import default_timer as timer

def rgb_conv(i):
    color = 255 * np.array( colorsys.hsv_to_rgb( i/200.0, 0.6, 0.8) )
    return tuple(color.astype(int))

start = timer()
@jit(target_backend='cuda')
def mandelbrot(x,y, iter):
    c=complex(x,y)
    z=0
    for i in range(0,iter):
        if (np.abs(z)>2):
            return rgb_conv(i)
        
        z = z**2+c
    
    return (100,100,100)
    

img = Image.new("RGB", (1000, 1000))
pixels = img.load()

width, height = img.size

xmin=0.37500012006186556
xmax=0.37500012006186989

ymin=-0.21663938843771133
ymax=-0.21663938843770689

fig, ax = plt.subplots(figsize=[8, 4])

n_img = 60
for x in range(0,width):

    print("%.2f %%" % (x / width* 100.0))
    for y in range(0,height):
        a = (xmax - xmin)/width
        b = xmin
        c = (ymax-ymin)/height
        d = ymin
        pixels[x,y] = mandelbrot(a*x+b, c*y+d, 10000000)


ax.imshow(img, origin='lower', extent=[xmin,xmax,ymin,ymax])
#ax2 = zoomed_inset_axes(ax, zoom=6, loc='lower left')
#ax2.imshow(img, origin='lower', extent=[-2.8,1.2,-1,1])

#x1,x2,y1,y2 = -1.79,-1.73,-0.025,0.025
#ax2.set_xlim(x1,x2)
#ax2.set_ylim(y1,y2)

#mark_inset(ax,ax2, loc1=2, loc2=4)

print(timer()-start)
#plt.savefig('anim1.png')
x = [0.3750001200618655]
y = [-0.2166393884377127]
plt.plot(x,y,marker="o", markerfacecolor='red')
plt.show()




        


