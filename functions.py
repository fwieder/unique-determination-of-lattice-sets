import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import time
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from operator import itemgetter
from itertools import combinations
from util import printProgressBar

## returns the Difference set of A as a multiset
def difference_multiset(A):
    DMA = []
    for i in range(len(A)):
        for j in range(len(A)):
            DMA.append(list(np.array(A[i])-np.array(A[j])))
    return DMA


## returns the multiset of all primitive vectors A
def primitive_multiset(A):
    M = difference_multiset(A)
    PMA = []
    for x in M:
        if np.gcd.reduce(x) == 1:
            PMA.append(x)
    return PMA


## plots all points of A if A is 2-dimensional
def set_plot(A):
    fig,ax = plt.subplots()
    x_len = makebounds(A)[0] + 1
    y_len = makebounds(A)[1] + 1
    size = max(x_len,y_len)
    major_ticks = np.arange(-size,size + 1,1)
    ax.set_xlim(-x_len,x_len)
    ax.set_ylim(-y_len,y_len)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    plt.grid(which= 'major', axis = 'both')
    plt.plot(np.array(A)[:,0], np.array(A)[:,1], 'ko')
    plt.show()
    plt.clf()
  
    
## plots all points of A if A is 3-dimensional
def set_plot_3d(A):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_len = makebounds(A)[0] + 1
    y_len = makebounds(A)[1] + 1
    z_len = makebounds(A)[2] + 1
    size = max(x_len,y_len,z_len)
    major_ticks = np.arange(-size,size +1,1)
    ax.set_xlim(-x_len,x_len)
    ax.set_ylim(-y_len,y_len)
    ax.set_zlim(-z_len,z_len)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.set_zticks(major_ticks)
    x =[A[i][0] for i in range(len(A))]
    y =[A[i][1] for i in range(len(A))]
    z =[A[i][2] for i in range(len(A))]
    ax.scatter(x, y, z, c='k', marker='o')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()
    plt.clf()


## plots all points of A and the convex hull of A
def hull_plot(A):
    fig,ax = plt.subplots()
    points = np.array(A)
    hull = ConvexHull(points)
    x_len = makebounds(A)[0] + 1
    y_len = makebounds(A)[1] + 1
    size = max(x_len,y_len)
    major_ticks = np.arange(-size,size + 1,1)
    ax.set_xlim(-x_len,x_len)
    ax.set_ylim(-y_len,y_len)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    plt.grid(which= 'major', axis = 'both')
    plt.plot(points[:,0], points[:,1], 'ko')
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k--')
    plt.show()
    plt.clf


## returns a set containing all 2-dimensional points in a (2k+1)x(2l+1) box around the origin
def box(bounds):
    if len(bounds) == 2:
        bx = bounds[0]
        by = bounds[1]
        return[(x,y) for x in range(-bx,bx+1) for y in range(-by,by+1)]
    
    if len(bounds) == 3:
        bx = bounds[0]
        by = bounds[1]
        bz = bounds[2]
        return[(x,y,z) for x in range(-bx,bx+1) for y in range(-by,by+1) for z in range(-bz,bz+1)]
    

## returns the maximal absolute values of coordinates in A
def makebounds(A):
    return [max(A, key = itemgetter(i))[i] for i in range(len(A[0]))]


## returns True if A is a convex lattice set, returns false if is not a convex lattice set, or if A is not full dimensional
def is_cls(A):
    hull = ConvexHull(np.array(A),qhull_options="QJ")
    equations = hull.equations
    tol = 1e-12
    bound = box(makebounds(A))
    for x in bound:
        if x not in A:
            vec = np.dot(equations[:,:-1], np.array(x).T)
            if all(vec < -equations[:,-1] + tol):
                return False
    return True


## generates a 2-dimensional set containing half the points in a box excluding the origin 
def halfbox(bounds):
    halfbox = []
    if len(bounds) == 2:
        for y in range(1,bounds[1]+1):
            halfbox.append((0,y))
        for x in range(1,bounds[0]+1):
            for y in range(-bounds[1],bounds[1]+1):
                halfbox.append((x,y))
    if len(bounds) == 3:
        for y in range(1,bounds[1]+1):
            halfbox.append((0,y,0))
        for x in range(1,bounds[0]+1):
            for y in range(-bounds[1],bounds[1]+1):
                halfbox.append((x,y,0))
        for z in range(1,bounds[2]+1):
            for x in range(-bounds[0],bounds[0]+1):
                for y in range(-bounds[1],bounds[1]+1):
                    halfbox.append((x,y,z))
    return halfbox

## fill sets that are only in half the box with the other half to be origin-symmetric
def fill_left_side(A):
    if len(A[0])==2:
        fullset=[(0,0)]
        for z in A:
            fullset.append(z)
            fullset.append((-z[0],-z[1]))
    if len(A[0])==3:
        fullset=[(0,0,0)]
        for z in A:
            fullset.append(z)
            fullset.append((-z[0],-z[1],-z[2]))
    return fullset

## return all origin-symmetric convex-lattice sets of cardinality 2k+1 in the box
def c_examples(card,box):
    
    if card % 2 == 0:
        print("There are no origin-symmetric sets of even cardinality")
        return False
    
    k = int((card - 1)/2)
    
    if len(box) == 2:
        os_box_card =((2*box[0]+1)*(2*box[1]+1)-1)/2
    if len(box) == 3:
        os_box_card = ((2*box[0]+1)*(2*box[1]+1)*(2*box[2]+1)-1)/2
    
    os_sets_card = scipy.special.comb(os_box_card,k,exact = True)
    
    if len(box) == 2:
        print("There are", os_sets_card, "origin-symmetric lattice sets with cardinality", card, "in the [",2*box[0]+1, "x", 2*box[1]+1, "] box.")
    if len(box) == 3:
        print("There are ", os_sets_card, " origin-symmetric lattice sets with cardinality ", card," in the [",2*box[0]+1,"x",2*box[1]+1,"x",2*box[2]+1,"] box")
    ## generate all lattice sets with points on  the right of the y-axis
    ## and the points that are on the y-axis, for y >= 0 
    
    os_sets = list(combinations(halfbox(box),k))
    
    ## fill these sets with the points on the left, to obtain origin-symmetric sets
    
    for i,A in enumerate(os_sets):
        os_sets[i] = fill_left_side(A)

    ## filter out the convex lattice sets of all the origin-symmetric sets
    
    os_cls = []
    start = time.perf_counter()
    print("Testing sets for convexity: " + str(len(os_sets)))
    for i,A in enumerate(os_sets):
        if i % 1000 == 0:
            printProgressBar(i, os_sets_card, starttime=start)             
        if is_cls(A):
            os_cls.append(A)
    print("Time finished: " + str(time.perf_counter()-start))
    
    print("\nThere are", len(os_cls), "origin-symmetric, convex lattice sets with cardinality", card,"in the box.")
    
    ## filter out all sets, that have sets with equal projection counts:
    
    pmultisets = []
    if len(box) == 2:
        for x in os_cls:
            pmultisets.append(sorted(primitive_multiset(x), key=lambda element: (element[0],element[1])))
    if len(box) == 3:
        for x in os_cls:
            pmultisets.append(sorted(primitive_multiset(x), key=lambda element: (element[0],element[1],element[2])))
     
    equals = []
    start=time.perf_counter()
    print("Comparing projection counts: ")
    for i,A in enumerate(os_cls):
        if i % 10 == 0:
            printProgressBar(i, len(os_cls), starttime=start)
        for j,B in enumerate(os_cls):
            if A != B and pmultisets[i] == pmultisets[j] and A not in equals:
                    equals.append(A)

    print("\nThere are", len(equals)," sets, that are not uniquely determined by their projection counts")
    ## Print and plot the sets
    for A in equals:
        print(A)
        if len(A[0]) == 2:
           hull_plot(A)
        if len(A[0]) == 3:
            set_plot_3d(A)
    return equals
