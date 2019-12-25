from functions import *
import time

## Sets that form the counterexample in the Gardner, Gronchi, Zong paper:

A = [(-2,0),(-1,2),(-1,1),(-1,0),(0,1),(0,0),(0,-1),(1,0),(1,-1),(1,-2),(2,0)]
B = [(-2,1),(-1,1),(-1,0),(0,2),(0,1),(0,0),(0,-1),(0,-2),(1,0),(1,-1),(2,-1)]
C = [(-2,2),(-1,1),(-1,0),(-1,-1),(0,1),(0,0),(0,-1),(1,1),(1,0),(1,-1),(2,-2)]




start = time.time()

c_examples(4,3,5,True)
## temp = c_examples_3d(2,2,1,6,True)

end = time.time()
