# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 16:33:05 2021

@author: fhekk
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

heatmap=np.load("C:/Users/fhekk/Downloads/heatmap.npy")
target=np.load("C:/Users/fhekk/Downloads/target.npy")
scene=np.load("C:/Users/fhekk/Downloads/scene_img.npy")

scene=scene.astype(int)
plt.imshow(scene)
plt.title('scene')
plt.axis('off')
plt.show()
pxl_size=100

i=0
heat_resized=np.zeros((1080,1920))
for y in range(0,scene.shape[1]-pxl_size,pxl_size):
    for x in range(0,scene.shape[0]-pxl_size,pxl_size):
        heat_resized[x:x+pxl_size,y:y+pxl_size]+=(heatmap[i]*np.ones((pxl_size,pxl_size)))
        i+=1
    x_size=scene.shape[0]-pxl_size-x
    heat_resized[x+pxl_size:,y:y+pxl_size]+=(heatmap[i]*np.ones((x_size,pxl_size)))
    i+=1
for x in range(0,scene.shape[0]-pxl_size,pxl_size):
    y_size=scene.shape[1]-pxl_size-y
    heat_resized[x:x+pxl_size,y+pxl_size:]+=(heatmap[i]*np.ones((pxl_size,y_size)))
    i+=1
heat_resized[x+pxl_size:,y+pxl_size:]+=(heatmap[i]*np.ones((x_size,y_size)))
i+=1



plt.imshow(scene,'gray')
plt.imshow(heat_resized,'jet',alpha=0.25)
plt.title('heatmap-scene')
plt.axis('off')
plt.show()
plt.imshow(heat_resized,'jet')
plt.title('heatmap')
plt.axis('off')
plt.show()
np.save("C:/Users/fhekk/Downloads/scene_heatmap.npy",scene)