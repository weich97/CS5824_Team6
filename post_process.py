"""
Author:     Richard Tan (rickytan@vt.edu)
Version:    2019.12.08
"""

# ------------------------------------------------------------------------- Imports
import numpy as np, os
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------- Global Variables
DATA_DIR = ".\data"
LABELS = ["lr=0.0005, lambda=0.2", "lr=0.005, lambda=2", "lr=0.005, lambda=0.2"]
LABELS_SINGULARITY = ["removed 9.38% points around singularity", "removed 21.8% points around singularity"]
LABELS_CONTOURS = ["training", ""]

# ------------------------------------------------------------------------- Divergence
"""
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".npz"):
        filepath = os.path.join(DATA_DIR, filename)
        print("loading %s"%filename)

        # load divergence
        data = np.load(filepath)
        train_hist = data['a'][()]
        dr = train_hist['delta_real']

        # plot divergence
        plt.plot(range(len(dr)),dr)

# show plot
plt.legend(LABELS)
plt.yscale('log')
plt.ylabel('mean divergence')
plt.xlabel('epoch')
plt.show()
"""

# ------------------------------------------------------------------------- Contours

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".npz"):
        filepath = os.path.join(DATA_DIR, filename)
        print("loading %s"%filename)

        # load contours
        data = np.load(filepath)
        train_hist = data['a'][()]

        # set up data
        training = train_hist['validate'][0]
        x = np.linspace(-16,16,32)
        y = np.linspace(-16,16,32)
        u = training[0,:,:,0]
        v = training[0,:,:,1]
        speed = np.sqrt(u*u + v*v)

        # plot data
        plt.streamplot(x,y,u,v,color=speed)
        plt.show()
        

# x = train_hist['validate'][0] # (1,32,32,3)
"""
fig, ax = plt.subplots()
ax.contour(x[0,:,:,0])
plt.show()
"""