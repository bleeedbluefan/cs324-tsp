#!/usr/bin/python3
# Hi there!

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def greedy( self,time_allowance=60.0 ):
		pass
	
	
	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
		
	def branchAndBound( self, time_allowance=60.0 ):
		pass



	''' <summary>
		2-OPT ALGORITHM
		Builds off of Greedy algorithm
		
		Details to think about:
			* how do we avoid checking paths we've already seen?
				- index values on linked lists, only check values 'ahead' of itself
				(this might work well with the city limit, easy way to only check certain values ahead of itself)
			* do we want to evaluate each path and then take the best one, or swap immediately when a better solution is found?
			* Storage: Singly Linked List, maybe a list for indexes?
			
		Variables:
		For class:
			* Final Best Tour
			* Best Solution cost
			* total solutions found (increases when swapped)
			* Time elapsed
			
		For us:
			* Current Tour
			* total swaps made
			* (boolean) improved
			* k value (how many cities away do we want to check?)
		
		[Limit: 10 cities away?]
		NEEDED FUNCTIONS:
		* are two paths swappable?
			- can x1 hook to y2 and can x2 hook to y1?
			- does a reverse path exist? Use function below
		* can the path be reversed? - Pass in k value for cities away
			- if the path cannot reverse, we can exit
			- we only want to check the smaller path (nodes between the new path)
		* what is the cost of the new path?
			- if the cost is ever more, we can exit
			(Sub function for this?)
			- compare the swapped fragment with the original fragment, not whole path
		* swap two paths
		
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy( self,time_allowance=60.0 ):
		'''

		while improved:
				for range of path:
					for range of k:
						evaluate 2 opt swap (see functions above)
							* does a path exist to this city?
							* does the other city have a path to complete the tour?
							* can the path between the two reversed?
							* is the cost better?
							* swap or don't swap

	'''
		pass
		



