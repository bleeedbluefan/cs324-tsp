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

    # region old
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
        cost_matrix = np.array(
            [[np.inf if i == j else cities[i].costTo(cities[j]) for j in range(ncities)] for i in range(ncities)])
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

    def branchAndBound(self, time_allowance=60.0):
        # generate an initial bssf with random tour
        initial_result = self.defaultRandomTour()

        cities = self._scenario.getCities()
        ncities = len(cities)
        start_time = time.time()
        bcsf = initial_result['cost']
        bssf = initial_result['soln']
        total_solutions = 0
        max_queue = 0
        states_created = 1
        pruned_states = 0

        # generates initial cost matrix
        # Time Complexity: n^2
        # Space Complexity: n^2
        cost_matrix = np.ndarray(shape=(ncities, ncities))
        for city in cities:
            for other_city in cities:
                cost_matrix[city._index, other_city._index] = city.costTo(other_city)
        cost_matrix, lb = self.reduceCostMatrix(cost_matrix, ncities)

        queue = []
        route = [cities[0]]
        heapq.heapify(queue)

        new_city_count = ncities - 1
        heapq.heappush(queue, (new_city_count, lb, 0, cost_matrix, route))

        # pull elements from queue
        # while loop itself has a time complexity of n!
        # Overall time complexity : O(n!n^3)
        # Overall space complexity : in a theoretical worst case the queue would have an upper bound of
        # requiring O(n!n^2) space
        while len(queue) > 0 and time.time() - start_time < time_allowance:
            if len(queue) > max_queue:
                max_queue = len(queue)

            element = heapq.heappop(queue)
            curr_row = element[2]
            curr_matrix = element[3]
            curr_lb = element[1]
            cities_left = element[0]
            curr_route = element[4]

            if curr_lb > bcsf:
                pruned_states = pruned_states + 1
                continue

            if cities_left == 0:
                if curr_lb < bcsf:
                    total_solutions = total_solutions + 1
                    bcsf = curr_lb
                    bssf = TSPSolution(curr_route)
                    continue
            # Time complexity : n
            for col in range(ncities):
                if cost_matrix[curr_row, col] != np.inf:
                    cost = curr_matrix[curr_row, col]
                    if curr_lb + cost >= bcsf:
                        continue
                    # Space complexity : each copy is n^2
                    new_matrix = np.copy(curr_matrix)
                    # Time complexity : n
                    new_matrix = self.takePath(new_matrix, curr_row, col, ncities)
                    # Time complexity : n^2
                    new_matrix, reduce_cost = self.reduceCostMatrix(new_matrix, ncities)
                    newlb = curr_lb + cost + reduce_cost
                    states_created = states_created + 1
                    if newlb >= bcsf:
                        pruned_states = pruned_states + 1
                    else:
                        new_route = curr_route.copy()
                        new_route.append(cities[col])
                        new_cities_left = cities_left - 1
                        # Time Complexity : logn
                        heapq.heappush(queue, (new_cities_left, newlb, col, new_matrix, new_route))

        results = {}
        end_time = time.time()
        results['cost'] = bcsf
        results['time'] = end_time - start_time
        results['count'] = total_solutions
        results['soln'] = bssf
        results['max'] = max_queue
        results['total'] = states_created
        results['pruned'] = pruned_states
        return results

    # Time complexity of function: n
    def takePath(self, cost_matrix, row, col, ncities):
        # set row to inf
        # Time complexity of n
        for i in range(ncities):
            cost_matrix[row, i] = np.inf
        # set col to inf
        # Time complexity of n
        for i in range(ncities):
            cost_matrix[i, col] = np.inf
        cost_matrix[col, row] = np.inf
        return cost_matrix

    # Time complexity of function : n^2
    def reduceCostMatrix(self, cost_matrix, ncities):
        # reduce rows
        reduce_cost = 0
        # Time complexity : n
        for row in range(ncities):
            curr_low = np.inf
            # Time complexity : n
            for col in range(ncities):
                if cost_matrix[row, col] < curr_low:
                    curr_low = cost_matrix[row, col]
            if curr_low != np.inf and curr_low > 0:
                # Time complexity : n
                for col in range(ncities):
                    if cost_matrix[row, col] != np.inf:
                        cost_matrix[row, col] = cost_matrix[row, col] - curr_low
                reduce_cost = reduce_cost + curr_low

        # ensure each column has a zero value
        # Time complexity : n
        for col in range(ncities):
            curr_low = np.inf
            # Time complexity : n
            for row in range(ncities):
                if cost_matrix[row, col] < curr_low:
                    curr_low = cost_matrix[row, col]
            if curr_low != np.inf and curr_low > 0:
                # Time complexity : n
                for row in range(ncities):
                    if cost_matrix[row, col] != np.inf:
                        cost_matrix[row, col] = cost_matrix[row, col] - curr_low
                reduce_cost = reduce_cost + curr_low
        return cost_matrix, reduce_cost

    # endregion
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
        linked_list = CLinkedList()  # [03124] - 5167.5
        linked_list.add_cities(init_path["soln"])
        og_cost = init_path['cost']
        results = {}
        ncities = len(cities)
        num_swaps, found_swaps = 0, 0
        cost_matrix = np.array(
            [[np.inf if i == j else cities[i].costTo(cities[j]) for j in range(ncities)] for i in range(ncities)])

        k = 50
        if k >= ncities:
            k = ncities - 1

        improved = True
        # change this to be implemented correctly. Stores the value of the start index so the program knows when to stop
        # O(n^2k^2) space:O(k) - swaps
        for _ in range(linked_list.size):
            if time.time() - start_time >= time_allowance: break
            improved = False
            # we may need to store the current size of linked list as a class field
            # also need a reference to the first element of the list
            curr_city = linked_list.head
            for i in range(linked_list.size):
                eval_city = curr_city.next
                if time.time() - start_time >= time_allowance: break
                # may have to look into this, we may want it to run 1 less times than k since we don't want to evaluate the element next to it
                swaps = {}
                for j in range(k - 1):
                    # breaks out if we're back to the start
                    eval_city = eval_city.next
                    # code for checking if this is a potential path:
                    s, e, cost = self.evaluate_swap(curr_city, eval_city, linked_list, cost_matrix)
                    if cost < 0:
                        swaps[cost] = {'s': s, 'e': e}
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
        results['pruned'] = f"Took: {num_swaps}/{found_swaps}; {bssf.cost / og_cost:0.5f}"
        return results

    def evaluate_swap(self, entry_node, last_swapped, linked_list, cost_matrix):
        def get_cost(nodeA, nodeB):  # [1,3,0,2,4]
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
        first_node = start_node.next  # Current City
        exit_node = goal_node.next
        while curr_node.index != goal_node.index:
            previous_node = curr_node
            curr_node = next_node
            next_node = curr_node.next
            curr_node.next = previous_node
        start_node.next = goal_node
        first_node.next = exit_node