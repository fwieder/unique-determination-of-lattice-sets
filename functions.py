import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from operator import itemgetter
from itertools import combinations
import scipy.special
from util import printProgressBar
import time

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
    x_len = max_x(A) + 1
    y_len = max_y(A) + 1
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
    x_len = max_x(A) + 1
    y_len = max_y(A) + 1
    z_len = max_z(A) + 1
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
    points = np.array(list(set(A)))
    hull = ConvexHull(points)
    x_len = max_x(A) + 1
    y_len = max_y(A) + 1
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
def box(k,l):
    box = []
    for i in range(-k,k+1):
        for j in range(-l,l+1):
            box.append((i,j))
    return box


## returns a set containing all 3-dimensional points in a (2k+1)x(2l+1)x(2m+1) box around the origin
def box_3d(k,l,m):
    box = []
    for x in range(-k,k+1):
        for y in range(-l,l+1):
            for z in range(-m,m+1):
                box.append((x,y,z))
    return box


## returns the absolute value of x-coordinates in A
def max_x(A):
    Max = max(A, key = itemgetter(0))[0]
    Min = min(A, key = itemgetter(0))[0]
    return max(Max, -Min)


## returns the absolute value of y-coordinates in A
def max_y(A):
    Max = max(A, key = itemgetter(1))[1]
    Min = min(A, key = itemgetter(1))[1]
    return max(Max, -Min)


## returns the absolute value of z-coordinates in A
def max_z(A):
    Max = max(A, key = itemgetter(2))[2]
    Min = min(A, key = itemgetter(2))[2]
    return max(Max, -Min)


## returns True if A is a convex lattice set, returns false if is not a convex lattice set
def is_cls(A):
    hull = ConvexHull(np.array(A),qhull_options="QJ")
    equations = hull.equations
    tol = 1e-12
    
    ## genearte a bounding box, in which we are looking for points that are in the convex hull of A, but not in the
    ## set A itself. If there is such a point, A is not a convex lattice set.
    if len(A[0]) == 2:
        bound = box(max_x(A),max_y(A))
    if len(A[0]) == 3:
        bound = box_3d(max_x(A),max_y(A),max_z(A))
    
    ## check for all points in the box if they are in he convex hull of A. Since ConvexHull.equations yields Ax <= -b, we, multiply the 
    ## the point with the first columns of equations and check it the entries are all smaller than -b plus a tolerance.
    for x in bound:
        if x not in A:
            vec = np.dot(equations[:,:-1],np.array(x).T)
            if all(vec < -equations[:,-1] + tol):
                return False
    return True


## generates a 2-dimensional set containing half the points in a box excluding the origin 
def halfbox(bx,by):
    obox =[]
    for y in range(1,by+1):
        obox.append((0,y))
    for x in range(1,bx+1):
        for y in range(-by,by+1):
            obox.append((x,y))
    return obox


## generates a 3-dimensional set containing half the points in a box excluding the origin
def halfbox_3d(bx,by,bz):
    obox = []
    for y in range(1,by +1):
        obox.append((0,y,0))
    for x in range(1,bx+1):
        for y in range(-by,by+1):
            obox.append((x,y,0))
    for z in range(1,bz+1):
        for x in range(-bx,bx+1):
            for y in range(-by,by+1):
                obox.append((x,y,z))
    return obox


## fill sets that are only in half the box with the other half to be origin-symmetric
def fill_left_side(x):
    if len(x[0])==2:
        y=[(0,0)]
        for z in x:
            y.append(z)
            y.append((-z[0],-z[1]))
    if len(x[0])==3:
        y=[(0,0,0)]
        for z in x:
            y.append(z)
            y.append((-z[0],-z[1],-z[2]))
    return y


## return all origin-symmetric convex-lattice sets in a [2bx+1]x[2by+1] box of cardinality 2k+1
def c_examples(bx,by,k,plot = False):
    card = 2*k+1
    os_bound = ((2*bx+1)*(2*by+1)-1)/2
    os_sets_card = scipy.special.comb(os_bound,k,exact = True)
    print("There are", os_sets_card, "origin-symmetric lattice sets with cardinality", card, "in the [",2*bx+1, "x", 2*by+1, "] box.")
    
    ## generate all lattice sets with points on  the right of the y-axis
    ## and the points that are on the y-axis, for y >= 0 
    
    
    os_sets = list(combinations(halfbox(bx,by),k))    
    
    ## fill these sets with the points on the left, to obtain origin-symmetric sets
    
    for i,x in enumerate(os_sets):
        os_sets[i] = fill_left_side(x)
    
    ## filter out the convex lattice sets of all the origin-symmetric sets
    
    os_cls = []
    start = time.perf_counter()
    print("Testing sets for convexity: ")
    for i,x in enumerate(os_sets):
        if i % 1000 == 0:
            printProgressBar(i, os_sets_card, starttime=start)             
        if is_cls(x):
            os_cls.append(x)
            
    
    print("\nThere are", len(os_cls), "origin-symmetric, convex lattice sets with cardinality", card,"in the[", 2*bx+1, "x", 2*by+1, "] box.")
    
    ## filter out all sets, that have sets with equal projection counts:
    
    pmultisets = []
    for x in os_cls:
        pmultisets.append(sorted(primitive_multiset(x), key=lambda element: (element[0],element[1])))
    
    equals = []
    start=time.perf_counter()
    print("Comparing projection counts: ")
    for i,x in enumerate(os_cls):
        printProgressBar(i, len(os_cls), starttime=start)
        for j,y in enumerate(os_cls):
            if x != y and pmultisets[i] == pmultisets[j] and x not in equals:
                    equals.append(x)

    print("\nThere are", len(equals)," sets, that are not uniquely determined by their projection counts")
    for x in equals:
        print(np.array(x).T)
    
    ## PLot the sets if plot is True
    
    if plot == True:
        for x in equals:
            hull_plot(x)
    return equals

def c_examples_3d(bx,by,bz,k,plot=False):
    card = 2*k +1
    os_bound = ((2*bx+1)*(2*by+1)*(2*bz+1)-1)/2
    os_sets_card = scipy.special.comb(os_bound,k,exact = True)
    
    print("There are ", os_sets_card, " origin-symmetric lattice sets with cardinality ", card," in the [",2*bx+1,"x",2*by+1,"x",2*bz+1,"] box")
    
    os_sets = list(combinations(halfbox_3d(bx,by,bz),k))

    for i,x in enumerate(os_sets):
        os_sets[i] = fill_left_side(x)

    os_cls = []
    start = time.perf_counter()

    print("Testing sets for convexity: ")
    for i, x in enumerate(os_sets):
        if i % 1000 == 0:
            printProgressBar(i, os_sets_card, starttime=start)    
        if is_cls(x):
            os_cls.append(x)
    
    print("\nThere are ", len(os_cls)," origin-symmetric, convex lattice sets with cardinality ",card," in the[",2*bx+1,"x",2*by+1,"x",2*bz+1,"] box.")
    
    pmultisets = []
    for x in os_cls:
        pmultisets.append(sorted(primitive_multiset(x), key=lambda element: (element[0],element[1],element[2])))
    
    equals = []
    start=time.perf_counter()
    print("Comparing projection counts: ")
    for i,x in enumerate(os_cls):
        printProgressBar(i, len(os_cls), starttime=start)
        for j,y in enumerate(os_cls):
            if x != y and pmultisets[i] == pmultisets[j] and x not in equals:
                    equals.append(x)
        
   
    print("\nThere are", len(equals)," sets, that are not uniquely determined by their projection counts")
    for x in equals:
        print(np.array(x).T)
    if plot == True:
        for x in equals:
            set_plot_3d(x)
    
    return equals
