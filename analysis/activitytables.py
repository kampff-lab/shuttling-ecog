# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 05:22:49 2014

@author: Gonçalo
"""

import os
import cv2
import glob
import video
import imgproc
import datetime
import numpy as np
import pandas as pd
import activitymovies
import scipy.stats as stats
from scipy.interpolate import interp1d
from preprocess import appendtrialinfo
from preprocess import gapslice, stepslice
from preprocess import storepath, labelpath
from preprocess import frontactivity_key, rewards_key, info_key
from preprocess import max_width_cm, width_pixel_to_cm, center_cm
from preprocess import rail_start_pixels, rail_stop_pixels
from preprocess import steprois_cm, gaprois_cm
from preprocess import steprois_crop, gaprois_pixels
from preprocess import rail_start_cm, rail_stop_cm

heightcutoff = 20.42
cropstart = str(rail_start_cm)
cropstop = str(rail_stop_cm)
cropleft = rail_start_pixels * width_pixel_to_cm
cropright = rail_stop_pixels * width_pixel_to_cm
heightfilter = str.format('yhead_max > 0 and yhead_max < {0}',heightcutoff)
positionfilter = str.format('xhead_min >= {0} and xhead_max <= {1}',
                            cropstart, cropstop)
speedfilter = 'xhead_speed_25 > 2'
ballisticquery = str.format('{0} and {1} and {2}',
                   heightfilter,positionfilter,speedfilter)

def _lesioncategorymap_(volume):
    if volume == 0:
        return 'control'
    elif volume < 15:
        return 'halflesion'
    elif volume > 50:
        return 'lesiondecorticate'
    else:
        return 'lesion'

def lesioncategory(info):
    lesionvolume = info['lesionleft'] + info['lesionright']
    category = lesionvolume.map(_lesioncategorymap_)
    category.name = 'category'
    return category

def groupbyname(data,info):
    result = data.copy(True)
    result['l2'] = ['individual']*len(data)
    result.reset_index(inplace=True)
    result.sort(['session','subject'],inplace=True)
    result.set_index(['session','l2','subject'],inplace=True)
    return result

def _charrange_(stop):
    s = ord('a')
    return [chr(s+i) for i in range(stop)]

def groupbylesionvolumes(data,info,rename=False):
    levsession = 'session' in data.index.names
    lesionvolume = info['lesionleft'] + info['lesionright']
    lesionvolume.name = 'lesionvolume'
    g = pd.concat([data,lesionvolume,info['cagemate']],axis=1)
    if g.index.names[0] is None:
        g.index.names = ['subject']
    #joininfo = pd.concat((lesionvolume,info['cagemate']),axis=1)
    #g = data.join(joininfo)
    lesionorder = g[g['lesionvolume'] > 0].sort('lesionvolume',ascending=False)
    controls = lesionorder.groupby('cagemate',sort=False).median().index
    controls.name = 'subject' # OPTIONAL?
    controlorder = g.reset_index().set_index('subject').ix[controls]
    if rename:
        lnames = pd.Series(['L'+c for c in _charrange_(len(lesionorder))])
        cnames = pd.Series(['C'+c for c in _charrange_(len(controlorder))])
        lnames.name = 'subject'
        cnames.name = 'subject'
        lesionorder.index = lnames
        controlorder.index = cnames
    if levsession:
        controlorder.set_index('session',append=True,inplace=True)
    
    result = pd.concat([controlorder,lesionorder])
    result['lesion'] = ['lesion' if v > 0 else 'control'
                        for v in result['lesionvolume']]
    result.reset_index(inplace=True)
    if levsession:
        result = result[~result.session.isnull()]
    columns = ['subject' if c == 'level_0' else c for c in result.columns]
    result.columns = columns
    if levsession:
        result.sort(['session','lesion'],inplace=True)
        result.set_index(['session','lesion','subject'],inplace=True)
    else:
        result.sort(['lesion'],inplace=True)
        result.set_index(['lesion','subject'],inplace=True)
    result.drop(['lesionvolume','cagemate'],axis=1,inplace=True)
    return result
    
def trialactivity(rr,lpoke,rpoke,cr,vcr):
    lpoketime = lpoke.groupby('trial').duration.sum().to_frame('poketime')
    rpoketime = rpoke.groupby('trial').duration.sum().to_frame('poketime')
    ptime = pd.concat([lpoketime,rpoketime]).sort()
    crtime = cr.groupby('trial').duration.sum().to_frame('crossingtime')
    vcrtime = vcr.groupby('trial').duration.sum()
    vtime = (vcrtime - crtime.crossingtime).to_frame('visibletime')
    ttime = rr.time.diff()[1:] / np.timedelta64(1,'s')
    ttime.columns = ['totaltime']
    trialact = pd.concat([ptime,vtime,crtime,ttime],
                         join='inner',axis=1)
    atime = trialact[['poketime','visibletime','crossingtime']].sum(axis=1)
    idletime = trialact.time - atime
    trialact.insert(3,'idletime',idletime)
    return trialact
    
def firstordefault(condition,default=None):
    indices = np.where(condition)[0]
    if len(indices) > 0:
        return condition.index[indices[0]]
    return default
    
def cumsumreset(data,reset):
    a = ~reset
    c = np.cumsum(a)
    d = np.diff(np.concatenate(([0.],c[reset])))
    v = data.copy()
    v.ix[reset] = -d
    return np.cumsum(v)
    
def utcfromdatetime64(dt64):
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return datetime.datetime.utcfromtimestamp(ts)
    
def getkeyloc(index,key):
    if np.iterable(key):
        return [index.get_loc(k) for k in key]
    return index.get_loc(key)

def geomediancost(median,xs):
    return np.linalg.norm(xs-median,axis=1).sum()
    
def zscore(xs):
    return (xs - xs.mean()) / xs.std()
    
def mediannorm(xs):
    return xs - xs.median()
    
def normalize(xs,func,column=None,by=None,level=None):
    if column is None:
        column = xs.columns
        
    data = xs[column]
    if (by is not None) or (level is not None):
        data = data.groupby(by=by,level=level,sort=False)
        
    xs[column] = data.apply(func)

def mad(xs):
    return mediannorm(xs).abs().median()
    
def flipleftwards(x,side,aligncenter=True):
    x = x.copy(deep=True)
    leftwards = side == 'leftwards'
    x[leftwards] = max_width_cm - x[leftwards]
    if aligncenter:
        x -= steprois_cm.center[3][1]
    return x
    
def joinstepactivity(steps,crossings,activity):
    crossings = crossings.rename(columns={'index':'crossing'})
    crossings.set_index('crossing',append=True,inplace=True)
    stepact = activity.ix[steps.reset_index('time')['time']]
    stepact.reset_index(inplace=True)
    stepact.set_index(['subject','session','crossing'],inplace=True)
    return stepact.join(crossings,rsuffix='crossing')
    
def normalizeposition(xs,inplace=False):
    pos = ['xhead','yhead']
    left = xs.side == 'leftwards'
    right = xs.side == 'rightwards'
    zlft = xs.loc[left,pos].groupby(level='subject',sort=False).apply(zscore)
    zrht = xs.loc[right,pos].groupby(level='subject',sort=False).apply(zscore)
    if not inplace:
        xs = xs.copy(deep=True)
    xs.loc[left,pos] = zlft.values
    xs.loc[right,pos] = zrht.values
    if not inplace:
        return xs
    
def read_activity(path):
    return pd.read_hdf(storepath(path), frontactivity_key)
    
def read_rewards(path):
    return pd.read_hdf(storepath(path), rewards_key)
    
def read_crossings(path, activity):
    crosses = crossings(activity)
    labelh5path = labelpath(path)
    if os.path.exists(labelh5path):
        crosses.label = pd.read_hdf(labelh5path, 'label')
    return crosses
    
def read_crossings_group(folders):
    crossings = []
    for path in folders:
        activity = read_activity(path)
        cr = read_crossings(path, activity)
        cr['session'] = os.path.split(path)[1]
        crossings.append(cr)
    return pd.concat(crossings)
    
def appendlabels(data,labelspath):
    if os.path.exists(labelspath):
        with open(labelspath) as f:
            for line in f:
                label,value = line.split(':')
                try:
                    value = float(value)
                except ValueError:
                    value = value
                data[label] = value
                
def findsessions(folder, days=None):
    sessionpaths = glob.glob(os.path.join(folder,'**/front_video.csv'))
    folders = (os.path.split(path)[0] for path in sessionpaths)
    folders = [path for path in folders if os.path.exists(storepath(path))]
    if days is not None:
        if type(days) is slice:
            folders = folders[days]
        elif np.iterable(days):
            folders = [folders[day] for day in days]
        else:
            folders = [folders[days]]
    return folders
    
def read_subjects(folders, days=None,
                  key=frontactivity_key, selector=None,includeinfokey=True):
    if isinstance(folders, str):
        folders = [folders]
                      
    subjects = []
    for path in folders:
        subject = read_sessions(findsessions(path, days),
                                key,selector,includeinfokey)
        subjects.append(subject)
    return pd.concat(subjects)
    
def read_sessions(folders, key=frontactivity_key, selector=None,
                  includeinfokey=True):
    if isinstance(folders, str):
        folders = [folders]
    
    multikey = not isinstance(key,str) & np.iterable(key)
    if multikey and selector is None:
        raise ValueError("A table selector has to be specified for multi-keys.")
    
    sessions = []
    for path in folders:
        if multikey:
            tables = [pd.read_hdf(storepath(path), k) for k in key]
            session = selector(*tables)
        else:
            session = pd.read_hdf(storepath(path), key)
            if selector is not None:
                session = selector(session)

        if len(session) > 0 and key != info_key and includeinfokey:
            info = pd.read_hdf(storepath(path), info_key)
            info.reset_index(inplace=True)
            keys = [n for n in session.index.names if n is not None]
            session.reset_index(inplace=True)
            session['subject'] = info.subject.iloc[0]
            session['session'] = info.session.iloc[0]
            session.set_index(['subject', 'session'], inplace=True)
            session.set_index(keys, append=True, inplace=True)
        sessions.append(session)
    return pd.concat(sessions)
    
def read_canonical(folders,days=None):
    act = read_subjects(folders,days)
    cr = read_subjects(folders,days,selector=fullcrossings)
    fcr = read_subjects(folders,days,selector=crossings)
    info = read_subjects(folders,days,key=info_key)
    return act,cr,fcr,info
    
def slowdown(crossings):
    return pd.DataFrame(
    [stats.linregress(crossings.entryspeed,crossings.exitspeed)],
     columns=['slope','intercept','r-value','p-value','stderr'])
     
def findpeaks(ts,thresh,axis=-1):
    if isinstance(ts,pd.Series):
        ts = ts.to_frame()    
    
    valid = ts > thresh if thresh > 0 else ts < thresh
    masked = np.ma.masked_where(valid,ts)

    views = np.rollaxis(masked,axis) if ts.ndim > 1 else [masked]
    clumpedpeaks = []
    for i,view in enumerate(views):
        clumped = np.ma.clump_masked(view)
        peaks = [ts[slce].ix[:,i].argmax() if thresh > 0 else ts[slce].ix[:,i].argmin()
                 for slce in clumped]
        clumpedpeaks.append(peaks)
    return clumpedpeaks if len(clumpedpeaks) > 1 else clumpedpeaks[0]
     
def roiactivations(roiactivity,thresh,roicenters):
    roidiff = roiactivity.diff()
    roipeaks = findpeaks(roidiff,thresh)
    data = [(peak,i,roicenters[i][1],roicenters[i][0])
            for i,step in enumerate(roipeaks)
            for peak in step]
    data = np.array(data)
    data = data[np.argsort(data[:,0]),:]
    return data
     
def steptimes(activity,thresh=1500):
    stepactivity = activity.iloc[:,stepslice]
    data = roiactivations(stepactivity,thresh,steprois_cm.center)
    index = pd.Series(data[:,0],name='time')
    return pd.DataFrame(data[:,1:],
                        index=index,
                        columns=['stepindex',
                                 'stepcenterx',
                                 'stepcentery'])
                                 
def sliptimes(activity,thresh=1500):
    gapactivity = activity.iloc[:,gapslice]
    data = roiactivations(gapactivity,thresh,gaprois_cm.center)
    index = pd.Series(data[:,0],name='time')
    return pd.DataFrame(data[:,1:],
                        index=index,
                        columns=['gapindex',
                                 'gapcenterx',
                                 'gapcentery'])

def _crossinterp_(cr,activity,xpoints,ypoints,selector=lambda x:x.yhead):
    key = cr.name + (int(cr['index']),)
    trial = activity.loc[key,:]
    xhead = trial.xhead
    yhead = selector(trial)
    if cr.side == 'leftwards':
        xhead = max_width_cm - xhead
    curve = interp1d(xhead,yhead,bounds_error=False)
    ypoints.append(curve(xpoints))
    return cr
                                 
def spatialinterp(xpoints,activity,crossings,selector=lambda x:x.yhead):
    ypoints = []
    crossings[['index','side']].apply(
        lambda cr:_crossinterp_(cr,activity,xpoints,ypoints,selector),
        axis=1)
    return np.array(ypoints)
    
def spatialaverage(xpoints,crossingactivity,column=None,baseline=None):
    ypoints = []
    trials = crossingactivity.groupby(level=['subject','session','crossing'])
    for key,trial in trials:
        if column is None:
            data = (trial.time - trial.time[0]) / np.timedelta64(1,'s')
        else:
            data = trial[column]
        curve = interp1d(trial.xhead,data,bounds_error=False)
        data = curve(xpoints).reshape(1,-1)
        if baseline is not None:
            data -= np.median(data[:,baseline])
        ypoints.append(data)        
    ypoints = np.concatenate(ypoints,axis=0)
    return np.mean(ypoints,axis=0),stats.sem(ypoints,axis=0)

def crossingspatialaverage(activity,crossings,selector=lambda x:x.yhead):
    ypoints = []
    xpoints = np.linspace(rail_start_cm,rail_stop_cm,100)
    for s,side in crossings[['timeslice','side']].values:
        trial = activity.xs(s,level='time')
        xhead = trial.xhead
        yhead = selector(trial)
        if side == 'leftwards':
            xhead = max_width_cm - xhead
        curve = interp1d(xhead,yhead,bounds_error=False)
        ypoints.append(curve(xpoints))
    ypoints = np.array(ypoints)
    return xpoints,np.mean(ypoints,axis=0),stats.sem(ypoints,axis=0)
    
def _getactivityslice_(activity,key,columns):
    if isinstance(activity.index, pd.core.index.MultiIndex):
        return activity.xs(key,level='time',
                           drop_level=False).ix[:,columns]
    else:
        return activity.ix[key,columns]

def getroipeaks(activity,roislice,trial,leftroi,rightroi,roiinfo,
                usediff=True,thresh=1500,headinfront=True):
    leftwards = trial.side == 'leftwards'
    roiindex = leftroi if leftwards else rightroi
    roiactivity = _getactivityslice_(activity,trial.timeslice,roislice)
        
    if usediff:
        roiactivity = roiactivity.diff()
    roipeaks = findpeaks(roiactivity,thresh)[roiindex]
    
#   This constraint checks if the head is BEFORE the next rail (????)
    if headinfront:
        roipeaks = [peak for peak in roipeaks
                     if (activity.xhead[peak] > roiinfo.min[leftroi-1][1] if leftwards else
                         activity.xhead[peak] < roiinfo.max[rightroi+1][1]).any()]
    return roipeaks
    
def getsteppeaks(activity,trial,leftstep,rightstep):
    return getroipeaks(activity,stepslice,trial,leftstep,rightstep,steprois_cm)
    
def getslippeaks(activity,trial,leftgap,rightgap):
    return getroipeaks(activity,gapslice,trial,leftgap,rightgap,gaprois_cm,
                       usediff=False,thresh=5000,headinfront=False)

def roicrossings(activity,crossings,leftroi,rightroi,getpeaks):
    indices = []
    
    for index,trial in crossings.iterrows():
        roipeaks = getpeaks(activity,trial,leftroi,rightroi)
        if len(roipeaks) > 0:
            indices.append(index)
    return crossings.loc[indices]

def stepcrossings(activity,crossings,leftstep,rightstep):
    return roicrossings(activity,crossings,leftstep,rightstep,getsteppeaks)
    
def slipcrossings(activity,crossings,leftgap,rightgap):
    return roicrossings(activity,crossings,leftgap,rightgap,getslippeaks)

def roiframeindices(activity,crossings,leftroi,rightroi,getpeaks):
    indices = []
    side = []
    
    for index,trial in crossings.iterrows():
        roipeaks = getpeaks(activity,trial,leftroi,rightroi)            
        if len(roipeaks) > 0:
            peakframes = [_getactivityslice_(activity,peak,'frame')
                          for peak in roipeaks]
            frameindex = min(peakframes)
            indices.append(frameindex)
            side.append(trial.side)
    return indices,side
    
def stepframeindices(activity,crossings,leftstep,rightstep):
    return roiframeindices(activity,crossings,leftstep,rightstep,getsteppeaks)
    
def slipframeindices(activity,crossings,leftgap,rightgap):
    return roiframeindices(activity,crossings,leftgap,rightgap,getslippeaks)
    
def stepfeature(activity,crossings,leftstep,rightstep):
    indices,side = stepframeindices(activity,crossings,leftstep,rightstep)
    features = activity.ix[indices,:]
    features['side'] = side
    return features
#    side = pd.DataFrame(side,columns=['side'])
#    side.index = features.index
#    return pd.concat((features,side),axis=1)

def croproi(frame,roiindex,roicenter_pixels,cropsize=(300,300),background=None,
            flip=False,cropoffset=(0,0)):
    roicenter = roicenter_pixels[roiindex]
    roicenter = (roicenter[0] + cropoffset[0], roicenter[1] + cropoffset[1])
    
    frame = imgproc.croprect(roicenter,cropsize,frame)
    if background is not None:
        background = imgproc.croprect(roicenter,cropsize,background)
        frame = cv2.subtract(frame,background)
    if flip:
        frame = cv2.flip(frame,1)
    return frame
    
def cropstep(frame,stepindex,cropsize=(300,300),background=None,flip=False):
    return croproi(frame,stepindex,steprois_crop.center,cropsize,background,flip)
    
def cropslip(frame,gapindex,cropsize=(300,300),background=None,flip=False):
    return croproi(frame,gapindex,gaprois_pixels.center,cropsize,background,flip,
                   cropoffset=(-100,0))

def roiframes(indices,side,info,leftroi,rightroi,croproi,
               cropsize=(300,300),subtractBackground=False):
    # Tile step frames    
    vidpaths = activitymovies.getmoviepath(info)
    timepaths = activitymovies.gettimepath(info)
    backpaths = activitymovies.getbackgroundpath(info)
    videos = [video.video(path,timepath) for path,timepath in zip(vidpaths,timepaths)]
    
    frames = []
    for frameindex,side in zip(indices,side):
        leftwards = side == 'leftwards'
        roiindex = leftroi if leftwards else rightroi
        
        frame = videos[0].frame(frameindex)
        background = None
        if subtractBackground:
            timestamp = videos[0].timestamps[frameindex]
            background = activitymovies.getbackground(backpaths[0],timestamp)
        frame = croproi(frame,roiindex,cropsize,background,roiindex == rightroi)
        frames.append(frame)
    return frames
    
def stepframes(activity,crossings,info,leftstep,rightstep,
               cropsize=(300,300),subtractBackground=False):
    indices,side = stepframeindices(activity,crossings,leftstep,rightstep)
    return roiframes(indices,side,info,leftstep,rightstep,
                     cropstep,cropsize,subtractBackground)
                     
#def slipframes(activity,crossings,info,leftgap,rightgap,
#               cropsize=(300,300),subtractBackground=False):
#    indices,side = slipframeindices(activity,crossings,leftgap,rightgap)
#    return roiframes(indices,side,info,leftgap,rightgap,
#                     cropslip,cropsize,subtractBackground)
                     
def slipactivity(activity):
    rowindex = []
    rows = []    
    
    roiactivity = activity.ix[:,25:32]
    roipeaks = findpeaks(roiactivity,5000)
    for gapindex,gap in enumerate(roipeaks):
        for slip in gap:
            gapcenter = gaprois_cm.center[gapindex]
            slipactivity = roiactivity.ix[slip,gapindex]
            rowindex.append(slip)
            rows.append((gapindex,gapcenter[0],gapcenter[1],slipactivity))
    rowindex = pd.MultiIndex.from_tuples(rowindex,names=activity.index.names)
    data = pd.DataFrame(rows,rowindex,
                        columns=['gapindex','xgap','ygap','peakactivity'])
    return data
    
def countslipevents(slipactivity):
    slipactivity = slipactivity.reset_index()
    return slipactivity.groupby(['subject','session',
                                 'trial','gapindex'])['gapindex'].count()
    
def coldist(xs,xcol1,xcol2,ycol1,ycol2):
    return np.sqrt((xs[xcol1]-xs[xcol2])**2 + (xs[ycol1]-xs[ycol2])**2)
    
def setlist(l,mask,val):
    for i,v in enumerate(mask):
        if v:
            l[i] = val
    
def slipfilter(slipactivity):
    criteria = True
    gapcriteria = (slipactivity.gapindex > 0) & (slipactivity.side == 'rightwards')
    gapcriteria |= (slipactivity.gapindex < 6) & (slipactivity.side == 'leftwards')
#    criteria &= gapcriteria
    criteria &= slipactivity.yhead < 15
    criteria &= slipactivity.peakactivity > 8000
    criteria &= abs(slipactivity.xtail-slipactivity.xgap) > 5
    criteria &= (slipactivity.gapindex > 0) & (slipactivity.gapindex < 6)
    return criteria
    
def slipframes(slipactivity,info,cropsize=(300,200),
               subtractBackground=False):
    vidpaths = activitymovies.getmoviepath(info)
    timepaths = activitymovies.gettimepath(info)
    backpaths = activitymovies.getbackgroundpath(info)
    videos = [video.video(path,timepath) for path,timepath in zip(vidpaths,timepaths)]

    frames = []    
    for index,trial in slipactivity.iterrows():
        rightwards = trial.side == 'rightwards'
        frame = videos[0].frame(trial.frame)
        background = None
        if subtractBackground:
            timestamp = videos[0].timestamps[trial.frame]
            background = activitymovies.getbackground(backpaths[0],timestamp)
        if cropsize is not None:
            frame = cropslip(frame,trial.gapindex,cropsize,background,rightwards)
        frames.append(frame)
    return frames
    
def __lickbouts__(licks,time):
    bouts = []
    lickcounts = []
    for i,s in enumerate(licks):
        if len(bouts) == 0:
            bouts.append(s)
            lickcounts.append(1)
        else:
            currbout = bouts[-1]
            ili = time[s.start] - time[currbout.stop]
            if ili > datetime.timedelta(seconds=1.5):
                bouts.append(s)
                lickcounts.append(1)
            else:
                bouts[-1] = slice(currbout.stop,s.stop)
                lickcounts[-1] += 1
    return bouts,lickcounts
    
def pokebouts(poke,rr):
#    baseline = poke.median()
#    thresh = baseline + poke.std()
    thresh = 400 # from actual threshold values
    masked = np.ma.masked_array(poke, poke > thresh)
    flat = np.ma.clump_unmasked(masked)
    licks = [slice(flat[i-1].stop-1,flat[i].start)
            for i in range(1,len(flat))]
    
    # Generate poke features
    time = poke.index
    bouts,lickcounts = __lickbouts__(licks,time)
    if len(bouts) == 0:
        return pd.DataFrame()
    trialinfo = appendtrialinfo(poke.reset_index('time').time,rr,[rr])
    if len(rr) == 0:
        rewardoffset = [1] * len(bouts)
    else:
        rewardoffset = [abs((time[s.stop-1]-trialinfo.time[s.stop-1]).total_seconds())
                       for s in bouts]
    trialinfo = pd.DataFrame([trialinfo.trial[s].max() + (1 if o < 1 else 0)
                             for o,s in zip(rewardoffset,bouts)],
                             columns=['trial'])
    timeslice = pd.DataFrame([slice(time[s.start],time[s.stop-1])
                             for s in bouts],columns=['timeslice'])
    duration = pd.DataFrame([(time[s.stop-1]-time[s.start]).total_seconds()
                            for s in bouts],
                            columns=['duration'])
    peak = pd.DataFrame([poke[s].max()[0] for s in bouts],columns=['peak'])
    bouts = pd.DataFrame(bouts,columns=['slices'])
    licks = pd.DataFrame(lickcounts,columns=['licks'])
    return pd.concat([bouts,
                      timeslice,
                      trialinfo,
                      duration,
                      licks,
                      peak],
                      axis=1)

def cropcrossings(x,slices,crop,center=max_width_cm / 2.0):
    def test_slice(s):
        return (x[s] > crop[0]) & (x[s] < crop[1])
    
    def crop_slice(s):
        valid_mask = test_slice(s).astype(np.int).values
        cross_bits = np.ma.clump_unmasked(np.ma.masked_equal(valid_mask,False))
        cross_bits = (slice(b.start+s.start,b.stop+s.start) for b in cross_bits)
        main_bit = next((b for b in cross_bits if _testcrossing_(x,b,center)),
                        None)
        return main_bit
    cropped = (crop_slice(s) for s in slices)
    return [s for s in cropped if s is not None]
    
def _testcrossing_(xhead,s,center):
    return xhead[s.start] > center and xhead[s.stop-1] < center \
        or xhead[s.start] < center and xhead[s.stop-1] > center
    
def _getcrossingslice_(xhead,midcross=True,crop=True,
                     center=max_width_cm / 2.0):
    # Generate trajectories and crossings
    crossings = np.ma.clump_unmasked(np.ma.masked_invalid(xhead))
    if midcross:
        crossings = [s for s in crossings if _testcrossing_(xhead,s,center)]
    if crop:
        crossings = cropcrossings(xhead,crossings,[cropleft,cropright])
    return crossings
    
def getballistictrials(crossings):
    return crossings.query(ballisticquery)
    
def getstepslice(activity,stepfeature,before=200,after=400):
    slices = []
    for i,(index, row) in enumerate(stepfeature.iterrows()):
        ix = activity.index.get_loc(index)
        ixslice = slice(ix-before,ix+after+1)
        actslice = activity.ix[ixslice]
        frameindices = range(ixslice.stop - ixslice.start)
        if len(actslice) == len(frameindices):
            actslice['side'] = row.side
            actslice['crossindex'] = i
            actslice['frameindex'] = frameindices
            slices.append(actslice)
        else:
            print "Dropped slice!"
    if len(slices) > 0:
        return pd.concat(slices)
    else:
        return pd.DataFrame()

def stepfeatures(activity,leftstep=4,rightstep=3):
    cr = crossings(activity)
    return stepfeature(activity,cr,leftstep,rightstep)
    
def stepslices(activity,before=200,after=400):
    sf = stepfeatures(activity)
    return getstepslice(activity,sf,before,after)
    
def biasedsteps(activity,before=200,after=400):
    sf = stepfeatures(activity)
    stableacc = pd.DataFrame(cumsumreset(sf.stepstate3,~sf.stepstate3))
    unstableacc = pd.DataFrame(cumsumreset(sf.stepstate3,sf.stepstate3))
    stableacc = sf[stableacc.stepstate3 <= -3]
    unstableacc = sf[unstableacc.stepstate3 <= -3]
    stf = getstepslice(activity,stableacc,before,after)    
    uf = getstepslice(activity,unstableacc,before,after)
    stf['bias'] = True
    uf['bias'] = False
    return pd.concat((stf,uf))
    
def posturebias(steps,n=3):
    stablebias = []
    unstablebias = []
    for key,sf in steps.groupby(level=['subject','session']):
        bstable = pd.DataFrame(cumsumreset(sf.stepstate3,~sf.stepstate3))
        stablebias.append(sf[(bstable >= n).shift().fillna(False).values])
        bunstable = pd.DataFrame(cumsumreset(~sf.stepstate3,sf.stepstate3))
        unstablebias.append(sf[(bunstable >= n).shift().fillna(False).values])
    return pd.concat(stablebias),pd.concat(unstablebias)
    
def compensation(activity):
    cr = crossings(activity)
    sffore = stepfeature(activity,cr,4,3)
    sfhind = stepfeature(activity,cr,1,6)
    sffore.reset_index(inplace=True)
    sfhind.reset_index(inplace=True)
    sffore.set_index('trial',inplace=True)
    sfhind.set_index('trial',inplace=True)
    return sffore.join(sfhind,
                       how='inner',
                       lsuffix='_fore',
                       rsuffix='_hind')
                       
def crossingoffset(activity,crossings,offset=20.0):
    data = []
    for s,side in crossings[['timeslice','side']].values:
        trial = activity.ix[s]
        xhead = trial.xhead
        if side == 'leftwards':
            point = max_width_cm - steprois_cm.center[4][1]
            xhead = max_width_cm - xhead
        else:
            point = steprois_cm.center[3][1]
        point += offset
        dist = np.abs(xhead - point)
        minact = activity.ix[np.argmin(dist)]
        minact.is_copy = False
        minact['side'] = side
        data.append(minact)
        
    return pd.DataFrame(data)
    
def spatialactivity(activity,offset=20.0,ballistic=True):
    cr = crossings(activity)
    if ballistic:
        cr = getballistictrials(cr)

    return crossingoffset(activity,crossings,offset)
    
def crossingactivity(activity,midcross=True,crop=True,
                     center=max_width_cm / 2.0):
    crossings = _getcrossingslice_(activity.xhead,midcross,crop,center)
    if len(crossings) == 0:
        return pd.DataFrame()

    crossingactivity = pd.concat([activity.ix[s,:] for s in crossings])
    crossingindex = [i for i,s in enumerate(crossings)
                     for x in range(s.stop-s.start)]
    crossingactivity['crossing'] = crossingindex
    crossingactivity.reset_index(inplace=True)
    crossingactivity.set_index(['crossing','time'],inplace=True)
    return crossingactivity

def visiblecrossings(activity):
    return fullcrossings(activity,midcross=False)

def fullcrossings(activity,midcross=True):
    return crossings(activity,midcross,False)

def crossings(activity,midcross=True,crop=True,center=center_cm):
    xhead = activity.xhead
    crossings = _getcrossingslice_(xhead,midcross,crop,center)
    if len(crossings) == 0:
        return pd.DataFrame()
        
    # Trial info
    trialinfo = pd.DataFrame([activity.iloc[s.start,1:8] for s in crossings])
    trialinfo.reset_index(inplace=True,drop=True)
    trialinfo['stepstate'] = np.bitwise_or(1 << 7, np.bitwise_or(
        np.left_shift(trialinfo['stepstate1'], 6), np.bitwise_or(
        np.left_shift(trialinfo['stepstate2'], 5), np.bitwise_or(
        np.left_shift(trialinfo['stepstate3'], 4), np.bitwise_or(
        np.left_shift(trialinfo['stepstate4'], 3), np.bitwise_or(
        np.left_shift(trialinfo['stepstate5'], 2), np.bitwise_or(
        np.left_shift(trialinfo['stepstate6'], 1), 1)))))))
    
    # Generate crossing features
    time = activity.index
    timeslice = pd.DataFrame([slice(time[s.start],time[s.stop-1])
                             for s in crossings],columns=['timeslice'])
    label = pd.DataFrame(['valid' for s in crossings],columns=['label'])
    position = pd.DataFrame([(activity.xhead[s].min(),activity.xhead[s].max())
                            for s in crossings],
                            columns=['xhead_min','xhead_max'])
    height = pd.DataFrame([activity.yhead[s].describe() for s in crossings])
    height.columns = 'yhead_' + height.columns
    height.columns = [c.replace('%','') for c in height.columns]
    speed = pd.DataFrame([activity.xhead_speed[s].abs().describe()
                         for s in crossings])
    speed.columns = 'xhead_speed_' + speed.columns
    speed.columns = [c.replace('%','') for c in speed.columns]
    duration = pd.DataFrame([(time[s.stop-1]-time[s.start]).total_seconds()
                            for s in crossings],
                            columns=['duration'])
    side = pd.DataFrame(['rightwards' if activity.xhead[s.start] < center
                        else 'leftwards'
                        for s in crossings], columns=['side'])
    crosstime = pd.DataFrame([firstordefault(activity.xhead[s] >= center,
                                             pd.NaT)
                             if activity.xhead[s.start] < center
                             else firstordefault(activity.xhead[s] <= center,
                                                 pd.NaT)
                             for s in crossings], columns=['crosstime'])
    
    # Slowdown
    xspeed = activity.xhead_speed
    entrydistance = (cropright - cropleft) / 3.0    
    entrypoints = [xhead[s] < (cropleft + entrydistance)
    if xhead[s.stop-1] > xhead[s.start]
    else xhead[s] > (cropright - entrydistance)
    for s in crossings]
    exitpoints = [xhead[s] > (cropright - entrydistance)
    if xhead[s.stop-1] > xhead[s.start]
    else xhead[s] < (cropleft + entrydistance)
    for s in crossings]
        
    entryspeed = pd.DataFrame([np.abs(xspeed[s][v].mean())
    for s,v in zip(crossings,entrypoints)],columns=['entryspeed'])
    crossingspeed = pd.DataFrame([np.abs(xspeed[s][~v & ~x].mean())
    for s,v,x in zip(crossings,entrypoints,exitpoints)],
    columns=['crossingspeed'])    
    exitspeed = pd.DataFrame([np.abs(xspeed[s][v].mean())
    for s,v in zip(crossings,exitpoints)],columns=['exitspeed'])
        
    # Steps
    steptimes = pd.DataFrame([[step[0] if len(step) > 0 else pd.NaT
                 for step in findpeaks(activity.ix[s,stepslice].diff(),1500)]
                 for s in crossings])
    steptimes.columns = [str.format('steptime{0}',i)
                         for i in xrange(len(steptimes.columns))]
    
    # Slips
    gapactivity = pd.DataFrame([activity.ix[s,gapslice].max() for s in crossings])
    gapactivity.columns = [str.format('maxgap{0}',i)
                          for i in xrange(len(gapactivity.columns))]
    
    crossings = pd.DataFrame(crossings,columns=['slices'])
    return pd.concat([crossings,
                      timeslice,
                      label,
                      trialinfo,
                      duration,
                      position,
                      height,
                      speed,
                      side,
                      crosstime,
                      steptimes,
                      entryspeed,
                      crossingspeed,
                      exitspeed,
                      gapactivity],
                      axis=1)
    