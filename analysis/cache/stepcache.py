# -*- coding: utf-8 -*-
"""
Created on Sun May 03 15:08:56 2015

@author: Gonçalo
"""

from activitytables import read_subjects
from activitytables import stepfeatures
from datapath import ecog, ecogcache
from datapath import stepfeatures_key

# Rebuild crossing cache
print "Rebuilding ecog step features..."
cr = read_subjects(ecog,selector=stepfeatures)
cr.to_hdf(ecogcache,stepfeatures_key)
