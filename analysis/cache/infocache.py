# -*- coding: utf-8 -*-
"""
Created on Sat May 02 22:09:50 2015

@author: Gonçalo
"""

from activitytables import read_subjects
from activitytables import info_key
from datapath import ecog, ecogcache
from datapath import ecoganalysis, ensurefolder

# Rebuild info cache
ensurefolder(ecoganalysis)
print "Rebuilding ecog session info..."
cr = read_subjects(ecog,key=info_key)
cr.to_hdf(ecogcache,info_key)
