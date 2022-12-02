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


# ----------Linked List Class ADDED BY JUSTIN-----------
class Node:
    def __init__(self, city_info):
        self.info = city_info
        self.next = None

    def return_info(self):
        return self.info


class CLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    # Juliana: Adds an entire list of cities to the linked list. Mainly used for testing in Greedy
    def add_cities(self, cities):
        for city in cities:
            self.add_city(city)

    def add_city(self, city_info):

        # Make a new node with city_info, attach the new node's next to the head (making it a circle LL)
        new_node = Node(city_info)

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
            print(self.head.info)    # If head exists, print it and move to the next node
            temp = temp.next
            # While temp is not None or not head, then print its info
            while temp is not self.head and temp is not None:
                print(temp.info)
                temp = temp.next

    # Juliana: Returns an array version of the linked list. Might be useful for converting to BSSF
    def return_cities_as_array(self):
        arr_of_cities = [self.head.info]
        temp = self.head.next
        while temp is not self.head:    # For every unique node in the list, add its info to the array
            arr_of_cities.append(temp.info)
            temp = temp.next
        return arr_of_cities    # Return array for of list

    def get_size(self):
        return self.size

class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
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
        cities = self._scenario.getCities()
        num_cities = len(cities)
        start_time = time.time()
        starting_column = 1
        original_column = starting_column

        # Creates the Matrix
        # >>> MEMORY IS O(n^2) -- creates a matrix nXn size
        cost_matrix = [[0 for i in range(num_cities)] for j in range(num_cities)]

        # Starts the path
        path = [starting_column]

        # Populates the cost matrix with proper values
        # >>> TIME IS O(n^2) -- populates matrix that is nXn size
        for i in range(num_cities):
            for j in range(num_cities):
                cost_matrix[i][j] = cities[i].costTo(cities[j])

        # Traverse through each city to find NN
        # >>> TIME IS O(n^2) -- Each iteration check the city you're at & finds best next (might be O(Log(n^2))
        for i in range(num_cities):
            min_value = np.inf
            current_index = -1 # incase no path exists, set to -1 for down the road

            # Goes through each city
            for j in range(num_cities):
                current_value = cost_matrix[starting_column][j]

                # If a cities value is less, save it
                if current_value < min_value and j != original_column:
                    min_value = current_value
                    current_index = j

            # Store the best city and update new city for next iteration
            path.append(current_index)
            starting_column = current_index

            # make any other path infinity to prevent going down it

            for j in range(num_cities):
                cost_matrix[j][starting_column] = np.inf




        # Info stored here
        city_path = []
        results = {}

        # Traversing through NN
        # >>> TIME IS O(n) -- Runs through the path and appends city path n times
        for i in range(len(path) - 1):

            # If any path is -1? This means no path exists
            if path[i] == -1:
                results['cost'] = np.inf
                results['time'] = time.time() - start_time
                results['count'] = None
                results['soln'] = None
                results['max'] = None
                results['total'] = None
                results['pruned'] = None
                return results

            # Update path
            city_path.append(cities[path[i]])



        # Store path & All the results
        # >>> TIME IS O(1) -- Saves route and stores results in constant time
        route = TSPSolution(city_path)
        end_time = time.time()
        results['cost'] = route.cost
        results['time'] = end_time - start_time
        results['count'] = None
        results['soln'] = route
        results['max'] = None
        results['total'] = None
        results['pruned'] = None

        # ---- CODE BY JULIANA: TESTING LINKED LIST ----
        linkedList = CLinkedList()
        linkedList.add_cities(city_path)
        linkedList.return_cities()
        print(linkedList.return_cities_as_array())
        return results
    pass

    #------------- END OF JUSTIN'S CODE-------------------

    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints: 
        max queue size, total number of states created, and number of pruned states.</returns> 
    '''

    def branchAndBound(self, time_allowance=60.0):
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
        pass
