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


# ----------Linked List Class ADDED BY JUSTIN-----------
class Node:
    def __init__(self, city_index):
        self.index = city_index
        self.next = None

    def return_info(self):
        return self.index


class CLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
        self.cost = np.inf

    # Juliana: Adds an entire list of cities to the linked list. Mainly used for testing in Greedy
    def add_cities(self, solution):
        self.cost = solution.cost
        for city in solution.route:
            self.add_city(city)

    def add_city(self, city):
        # Make a new node with city_info, attach the new node's next to the head (making it a circle LL)
        new_node = Node(city._index)

        # This will add the node to tail of LL
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
            self.tail.next = self.head
        self.size += 1

    def get_head(self) -> Node:
        return self.head

    def return_cities(self):
        temp = self.head
        if self.head is not None:
            print(self.head.index)  # If head exists, print it and move to the next node
            temp = temp.next
            # While temp is not None or not head, then print its info
            while temp is not self.head and temp is not None:
                print(temp.index)
                temp = temp.next

    # Juliana: Returns an array version of the linked list. Might be useful for converting to BSSF
    def return_solution(self, cities):
        arr_of_cities = [cities[self.head.index]]
        temp = self.head.next
        while temp is not self.head:  # For every unique node in the list, add its info to the array
            arr_of_cities.append(cities[temp.index])
            temp = temp.next
        return TSPSolution(arr_of_cities)  # Return array for of list

    def get_size(self):
        return self.size


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario
#region old
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

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
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

    def greedy(self, time_allowance=60.0):

        # ----------- NEW CODE ADDED BY JUSTIN MCKEEN (Nearest Neighbor)------------

        # Gathering initial info, starting time, and picking starting column (city)
        # >> TIME IS O(n) -- gathering cities takes n time
        '''Initialize the cites from the scenario'''
        results, cities = {}, self._scenario.getCities()
        '''Initialize some of the data we need for results'''
        ncities, count, bssf = len(cities), 0, None
        cost_matrix = np.array([[np.inf if i == j else cities[i].costTo(cities[j]) for j in range(ncities)] for i in range(ncities)])
        start_time = time.time()
        ''' Random Permutation for the possible starting indexes;
			 this allows multiple NN searches to be run with no overlaps
				 Time: O(n) Space: O(n)
			 '''
        for k in np.random.permutation(ncities):
            '''Terminate if we are taking too long'''
            if time.time() - start_time >= time_allowance: break
            '''Create a list of city indexes to search through'''
            unvisited_cities = list(range(ncities))
            '''Initialize the starting route to random starting index and the from_city to the respective city'''
            from_city, route = k, [unvisited_cities.pop(k)]
            '''Search until all cities have been visited
					Time: O(n) Space: O(n)'''
            while unvisited_cities and time.time() - start_time < time_allowance:
                '''Finds the index in the unvisited_cities list(a list of indexes) of the least cost city to travel to.
						Time: O(n) Space: O(n)'''
                idx = min(range(len(unvisited_cities)), key=lambda i: cost_matrix[from_city][unvisited_cities[i]])
                '''Gets the actual city to travel to, rather than it's index'''
                to_city = unvisited_cities[idx]
                '''Terminate the visit to the city if it is impossible (cost is infinity)'''
                if cost_matrix[from_city][to_city] == np.inf: break
                '''Add the city to the route and remove it from the unvisited_cities'''
                route.append(unvisited_cities.pop(idx))
                '''Update the city to start the next search from'''
                from_city = to_city
            '''We have found a possible solution'''
            count += 1
            '''If we have visitied all the cities, and there is a path back, update and return the found solution'''
            if not unvisited_cities and cost_matrix[route[-1]][route[0]] != np.inf:
                '''Create a TSPSolution form the converted route(list of indexes) to actual cities'''
                bssf = TSPSolution([cities[i] for i in route])
                break
        end_time = time.time()
        '''Return values of results'''
        results['cost'] = bssf.cost if bssf is not None else np.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    # ------------- END OF JUSTIN'S CODE-------------------

    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints: 
        max queue size, total number of states created, and number of pruned states.</returns> 
    '''

    class State:
        """
		A class to hold each sub-state of the Branch and Bound Algorithm.
		Store a cost_matrix of costs, a route, a list of remaining cities, and the current cost.
		The cities themselves are not stored within the State class, just their indexes;
		this was to hopefully make it faster.

		Complexity:
			Init Time: O(n^2)
			Copy Time: O(n^2)
			Update Time: O(n^2)
			Space: O(n^2)
		"""

        def __init__(self, init_cities):
            """
			Creates the initial State of the Branch and Bound Algorithm and reduces it to find the base cost.
			Complexity:
				Time: O(n^2)
				Space: O(n^2)
			:param init_cities: List of cities to form cost_matrix from
			"""
            self.num_cities = len(init_cities)
            ''' Create a new cost_matrix, each cell is filled with cost to move from city[row] -> city[column]
				The diagonals are marked as infinity so cities do not return to themselves
					Time: O(n^2) Space: O(n^2) '''
            self.cost_matrix = \
                np.array([[np.inf if i == j else init_cities[i].costTo(init_cities[j]) \
                           for j in range(self.num_cities)] for i in range(self.num_cities)])
            ''' Initialize other default values '''
            self.last_city_idx = 0
            self.route = [0]
            ''' Create list of unvisited cites indexes with all but 0 '''
            self.unvisited_cities = [i for i in range(1, self.num_cities)]
            ''' Reduce the cost matrix to zero out minimum values '''
            self.cost = self.reduce_cost_matrix()

        def update_route(self, idx):
            """
			This function is used to update the route by updating the route of the State.
			This involves re-reducing the matrix, and adding it to it's cost
			Complexity:
				Time: O(n^2)
				Space: O(n)

			:param idx: Index of city to move to; matches the corresponding row/col index of the cost matrix
			"""
            ''' Add the cost to travel to the city to '''
            self.cost += self.cost_matrix[self.last_city_idx][idx]
            ''' Remove the return edge from the matrix '''
            self.cost_matrix[idx][self.last_city_idx] = np.inf
            ''' Remove the column that was visited '''
            self.cost_matrix[:, idx] = np.inf
            ''' Remove the row that was visited '''
            self.cost_matrix[self.last_city_idx] = np.inf
            ''' update the last visited city '''
            self.last_city_idx = idx
            ''' Add the idx to the route '''
            self.route.append(idx)
            self.unvisited_cities.remove(idx)
            ''' Reduce the cost matrix back to zeroed values
					 Time: O(n^2) Space: O(n^2) '''
            self.cost += self.reduce_cost_matrix()

        def reduce_cost_matrix(self):
            """
			Reduces all the rows in the cost matrix by the minimum value, then reduces the columns.
			This reduction skips rows that have a 0 or infinity as their minimum
			Complexity:
				Time: O(n^2)
				Space: O(n)

			:return: The base cost used to travel to any available city in the matrx
			"""

            def reduction(axis):
                """
				Small helper function, reduces every row/col in the matrix by the minimum value of the row.
				Complexity:
					Time: O(n^2)
					Space: O(n)

				:param axis: The axis to reduce on, according to numpy's axis system
				:return: The sum cost removed by the reduction
				"""
                ''' Use numpy to find minimum value in all rows/cols 
						Time: O(n) Space: O(n)'''
                min_vec = np.amin(self.cost_matrix, axis=axis)
                reduce_cost = 0
                ''' Go through every value in the minimum values vector
						Time: O(n) Space: O(1)'''
                for i, c in enumerate(min_vec):
                    if c == 0 or c == np.inf: continue
                    reduce_cost += c
                    ''' subtract the minimum value from the row/col '''
                    if axis == 1:
                        self.cost_matrix[i] -= c
                    else:
                        self.cost_matrix[:, i] -= c
                ''' return the sum of all minimum values '''
                return reduce_cost

            ''' Return cost of the row reduction, followed by col reduction '''
            return reduction(axis=1) + reduction(axis=0)

        def get_next_state(self, idx):
            """
			Creates a copy of the state, then moves it to the next city.
			None is returned if the move is invalid
			Complexity:
				Time: O(n^2) # None case is O(1)
				Space: O(n^2) # None case is O(1)

			:param idx: Index of the city to move the copied state to
			:return: None if no path is possible, otherwise a new State with an updated cost_matrix and route
			"""
            ''' Terminate and return None if the city is unreachable '''
            if self.cost_matrix[self.last_city_idx][idx] == np.inf: return
            ''' Create a deepcopy of the state to modify '''
            from copy import deepcopy
            next_state = deepcopy(self)
            ''' Update the new state to the next city and return it '''
            next_state.update_route(idx)
            return next_state

        def __lt__(self, other):  # heapq uses < operator to push and pop right one
            """
			Built in less than comparison; used to sort the heapq heap.
			Complexity:
				Time: O(1)
				Space: O(1)
			:param other: The other state to compare against
			:return: Boolean value of whether the state is less than the other state
			"""
            ''' Check if the number of cities left is identical, if it is compare the costs
				ones with fewer cities to visit are prioritized '''
            if len(self.unvisited_cities) == len(other.unvisited_cities):
                return self.cost < other.cost
            else:
                return len(self.unvisited_cities) < len(other.unvisited_cities)

    def branchAndBound(self, time_allowance=60.0):
        pass

    #endregion
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

    def fancy(self, time_allowance=60.0):
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

        start_time = time.time()

        init_path = self.greedy(time_allowance)
        if init_path['soln'] is None or init_path['soln'].cost == np.inf:
            init_path = self.defaultRandomTour(time_allowance)
        cities = self._scenario.getCities()
        # init_path = {'cost': 7230, 'time': 0.0, 'count': 6, 'soln': TSPSolution([cities[i] for i in [4,1,2,0,3]]), 'max': None, 'total': None, 'pruned': None}
        linked_list = CLinkedList() # [03124] - 5167.5
        linked_list.add_cities(init_path["soln"])
        og_cost = init_path['cost']
        results = {}
        ncities = len(cities)
        num_swaps, found_swaps = 0, 0
        cost_matrix = np.array([[np.inf if i == j else cities[i].costTo(cities[j]) for j in range(ncities)] for i in range(ncities)])

        k = 50
        if k >= ncities:
            k = ncities - 1

        improved = True
        # change this to be implemented correctly. Stores the value of the start index so the program knows when to stop
        # O(n^2k^2) space:O(k) - swaps
        for _ in range(linked_list.size):
            if time.time()-start_time >= time_allowance: break
            improved = False
            # we may need to store the current size of linked list as a class field
            # also need a reference to the first element of the list
            curr_city = linked_list.head
            for i in range(linked_list.size):
                eval_city = curr_city.next
                if time.time()-start_time >= time_allowance: break
                # may have to look into this, we may want it to run 1 less times than k since we don't want to evaluate the element next to it
                swaps = {}
                for j in range(k - 1):
                    # breaks out if we're back to the start
                    eval_city = eval_city.next
                    # code for checking if this is a potential path:
                    s, e, cost = self.evaluate_swap(curr_city, eval_city, linked_list, cost_matrix)
                    if cost < 0:
                        swaps[cost] = {'s':s, 'e':e}
                if len(swaps) > 0:
                    best = min(swaps.keys())
                    found_swaps += len(swaps)
                    if best < 0:
                        linked_list.cost += best
                        self.swap(swaps[best]['s'], swaps[best]['e'])
                        num_swaps += 1
                        improved = True
                # set linked list node to the next one in list
                curr_city = curr_city.next
            if not improved: break
        bssf = linked_list.return_solution(cities)
        end_time = time.time()
        '''Return values of results'''
        results['cost'] = bssf.cost if bssf is not None else np.inf
        results['time'] = end_time - start_time
        results['count'] = found_swaps
        results['soln'] = bssf
        results['max'] = og_cost
        results['total'] = linked_list.cost
        results['pruned'] = f"Took: {num_swaps}/{found_swaps}; {bssf.cost/og_cost:0.5f}"
        return results

    def evaluate_swap(self, entry_node, last_swapped, linked_list, cost_matrix):
        def get_cost(nodeA, nodeB): # [1,3,0,2,4]
            return cost_matrix[nodeA.index, nodeB.index]

        first_swapped = entry_node.next
        exit_node = last_swapped.next
        reversed_backward_cost = get_cost(entry_node, last_swapped) + get_cost(first_swapped, exit_node)
        if reversed_backward_cost == np.inf:
            return entry_node, last_swapped, np.inf
        partial_forward_cost = get_cost(entry_node, first_swapped) + get_cost(last_swapped, exit_node)
        search_node = first_swapped
        while search_node.index != last_swapped.index:
            reversed_backward_cost += get_cost(search_node.next, search_node)
            partial_forward_cost += get_cost(search_node, search_node.next)
            if reversed_backward_cost == np.inf: break
            search_node = search_node.next
        return entry_node, last_swapped, (reversed_backward_cost - partial_forward_cost)

    def swap(self, start_node, goal_node):
        curr_node = start_node.next
        next_node = curr_node.next
        first_node = start_node.next    # Current City
        exit_node = goal_node.next
        while curr_node.index != goal_node.index:
            previous_node = curr_node
            curr_node = next_node
            next_node = curr_node.next
            curr_node.next = previous_node
        start_node.next = goal_node
        first_node.next = exit_node
