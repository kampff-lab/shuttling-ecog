# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 11:20:49 2015

@author: Gonçalo
"""

import os
import pandas as pd

_ecogpath_ = r'D:/Protocols/Shuttling/ECoG'

ecogdata = os.path.join(_ecogpath_,'Data')
ecoganalysis = os.path.join(_ecogpath_,'Analysis')

ecogcache = os.path.join(ecoganalysis,'cache.hdf5')
crossingactivity_stable_key = 'crossingactivity_stable'
crossingactivity_unstable_key = 'crossingactivity_unstable'
crossingactivity_restable_key = 'crossingactivity_restable'
crossingactivity_random_key = 'crossingactivity_random'
crossingactivity_challenge_key = 'crossingactivity_challenge'
visiblecrossings_key = 'visiblecrossings'
fullcrossings_key = 'fullcrossings'
crossings_key = 'crossings'
stepfeatures_key = 'stepfeatures'
leftpokebouts_key = 'task/poke/left/pokebouts'
rightpokebouts_key = 'task/poke/right/pokebouts'

ecog = [os.path.join(ecogdata,'JPAK_74'),
        os.path.join(ecogdata,'JPAK_75'),
        os.path.join(ecogdata,'JPAK_84')]
               
def ensurefolder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)        

def _findsubjectpath_(name,subjects):
    result = filter(lambda x: os.path.split(x)[1] == name,subjects)
    if len(result) > 0:
        return result[0]
    else:
        return None
               
def subjectpath(name):
    return _findsubjectpath_(name,ecog)

def relativepath(info,path):
    return os.path.join(subjectpath(info.name[0]),info.dirname,path)

def sessionpath(info,path=''):
    return pd.Series(info.reset_index().apply(
        lambda x:os.path.join(subjectpath(x.subject), x.dirname, path),
        axis=1).values,
        index = info.index,
        name='path')