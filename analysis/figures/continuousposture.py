# -*- coding: utf-8 -*-
"""
Created on Mon Aug 03 14:06:38 2015

@author: Gonçalo
"""

import pandas as pd
import matplotlib.pyplot as plt
from activitytables import flipleftwards, info_key
from datapath import ecogcache, stepfeatures_key

# Load data
info = pd.read_hdf(ecogcache, info_key)
steps = pd.read_hdf(ecogcache,stepfeatures_key)
steps.xhead = flipleftwards(steps.xhead,steps.side)

# Plot data
fig, axes = plt.subplots(3,1)
for i,(s,d) in enumerate(steps.groupby(level='subject')):
    ax = axes[i]
    ticks = list(d.stepstate3.diff().nonzero()[0])
#    ticks = ticks[1:4] + [ticks[-1]]
    d.xhead.plot(ax=ax,
                 style='.',
                 title=s,
                 xticks=ticks)
    ax.set_xticklabels(['x' for ui in range(len(ticks))])
#    ax.set_ylim(0,8)
    ax.set_xlabel('trial')
    ax.set_ylabel('nose x (cm)')
plt.suptitle('step posture')
plt.tight_layout()
plt.show()

# Save plot
