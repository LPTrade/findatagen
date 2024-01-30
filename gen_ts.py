# generate time series
# for I interval 
default_t = 0.005
epoch = 100
start = 0.3
multivariate = 2


import random
random.seed()#(a=42)
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

def neightbours(y, thresh=default_t):
	return random.choices([ delta for delta in np.linspace(y -thresh, y+thresh)], k=1)

def sample(epoch=epoch,y=None):
	if not y :
		y= random.random() # start in [0, 1] uniform
	out = []
	for x in range(epoch):
		# compute random threshold within allowed range
		thresh = random.random()*min(y/10,default_t)
		y = neightbours(y, thresh=thresh) # draw a neighbours list
		y= y[0] # select next value 
		out.append(y)
	return out	


def minmax(distrib,begin=0):
    """
    Recursively search for local extremun 
    """
    distrib = np.array(distrib)
    if len(distrib) < 2 : return []
    mini_ind = np.argsort(distrib, axis=0)[:40]
    maxi_ind = np.argsort(distrib,axis=0)[::-1][:40] #translate indexes when recrusively called
    #d_distrib = distrib[1:]-distrib[:-1]
    #d2_distrib = d_distrib[1:]-d_distrib[:-1]
    #maxi_ind = np.argsort(d2_distrib)[::-1][:10]
    #maxi_ind +=2
    maxi_ind = merge(maxi_ind, distrib)
    mini_ind = merge(mini_ind, distrib, cmp=lowerthan)
    maxi = distrib[maxi_ind]
    mini = distrib[mini_ind]
    
    ax.scatter(maxi_ind, maxi,marker='v')
    ax.scatter(mini_ind,mini,marker='^')
    return (mini_ind, maxi_ind)

def greaterthan(x,y)   : return x > y

def lowerthan(x,y) : return x< y 
    
def merge(ind,val,cmp=greaterthan):
       """
       Merge local extremums indexes of a function
       Return the reduced list of extremums in w the appropriate compare function
       """
       # output list reduced of extremums indexes, initialized empty
       clusters=[]
       # sort indexes list of extremums
       ind = np.sort(ind,axis=0)
       # lookup value buffer
       val = val[ ind ]
       # index of current lookup value, initialized at the index of the first extremum 
       point=ind[0]
       # cluster extremum buffer, initialized at the value of the first extremum
       maxi=val[0]
       # cluster extremum index buffer
       maxind=ind[0]
       
       # look through index table
       for i in range(1,len(ind)) :
            # current index is locally close to last lookup value
            if ind[i]-point < 3:
            	# swap extremum info for current cluster buffer
            	if cmp(val[i],maxi) : 
            	    maxi = val[i]
            	    maxind=ind[i]       	  
            else :
            	# create new cluster
            	clusters.append(maxind)
            	maxi=val[i]
            	maxind=ind[i]
            # update lookup value 
            point = ind[i]
            
       if maxind > 0:
            clusters.append(maxind)
            
       return clusters
    


def plotind(min_ind, max_ind,distrib,k=0,reject=[]):
	# data structure for extremum values
	# TODO : use dataclasse generic class
	# following block of code merge extremum in pairs of (min, max)
	# thr " 
	color=[f'C{i}' for i in range(20)][k]
	ext_ind = [ (i,'min') for i in min_ind ]
	ext_ind += [ (i,'max') for i in max_ind ]
	# sort indexes of extremums 
	# ex : [ (3, 'min'), (7, 'max'), (15, 'max') , ... ]
	
	ext_ind =sorted(ext_ind, key=lambda x: x[0])
	"""
	from operator import itemgetter
	tmp = []
	for head, tail in zip(ext_ind[:-1], ext_ind[1:]):
		if head[1] == 'min' and tail[1] == 'min':
			head = min(head, tail,key=itemgetter(1))
			tmp.append(head)
	
	ext_ind = tmp
	"""
	# pariwise helper function
	# create successive 2-uplets for a list of elements 
	# [1,2,4, ...] -> (1,2), (2,4),...
	from itertools import pairwise
	pairs=[]
	
	# TODO : combine 
	# "graph-like" reduction of local minimums/maximums
	# if a = pred(b) "predecesseur"
	# if a and b are local minimums of f  
	# (a,b) -> min(a,b)
	# 
	#
	
	# filter pairs of (min,max) extremum (2-uplets)
	# reject indexes already assigned
	def reject_pair(p):
		return p in reject
	for i,j in pairwise(ext_ind):
		if i[1] == 'min' and j[1] == 'max':
			pairs.append((i[0],j[0]))
	#pairs = [ p for p in  pairs if not reject_pair(p) ]
	growth = 0
	old = 0
	linear_dist =[]	
	acc_dist = []
	acc = 0
	eps= 0.001
	
	# Search throught each pairs of extremum in (min,max) format
	for i,j in pairs:
			deriv = (distrib[j] - distrib[i])
			linear_dist += [ 0 for _ in range(old,i)]
			acc_dist += [  growth+eps for _ in range(old,i)]
			avg = [ deriv*(pad/(j-i)) for pad in range(j-i)]
			linear_dist += avg
			acc_dist +=[ a/distrib[i]+growth for a in avg]
			old = j
			
			growth += (distrib[j] - distrib[i])/distrib[i]
			ax.fill_between(list(range(i,j)), 0, distrib[i:j], alpha=0.4, fc=color)
			plt.text(i,distrib[i]/2, f'{deriv/distrib[i]*100:0.1f}%')
	if j:
		for _ in range(j,len(distrib)):
			linear_dist += [ 0 ]
			acc_dist += [growth]
	# profit accumulation
	ax.plot(linear_dist)
	ax.plot(acc_dist)				
	# find (i,j) such as ext_ind[i] is min, ext_ind[j] is max and i < j 
	# ensure there is no available k in ext_ind such k < i and k < j
	closing =(distrib[-1]-distrib[0])/distrib[0]
	plt.text(0.5,k*0.05, f'{growth*100:0.3f}%')
	plt.text(0.,k*0.05+0.025, f'closing value : {closing*100:0.1f}%')
	return pairs
"""
Plot multiple distributions on a single plane 
main loop :
	1) generate distribution
	2) plot distribution
	3) find extremum values
	4) plot tuples (min,max) 
	     and derivatives
"""
distrib = sample(y=start)
ax.plot(distrib)
min_ind, max_ind = minmax(distrib)
pairs = plotind(min_ind, max_ind,distrib)
for k in range(1,multivariate):
    distrib=sample(y=start)#y=distrib[-1])
    # draw the generated time-serie on the canvas 
    ax.plot(distrib)  
    # search and draw extermums 
    min_ind, max_ind = minmax(distrib)
    reject = [ ind for i,j in pairs for ind in range(i,j)]
    plotind(min_ind, max_ind,distrib,k=k, reject=reject)
# display the canvas 
plt.show()
