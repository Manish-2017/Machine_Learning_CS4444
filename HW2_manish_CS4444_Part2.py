#Code Source: github.com/aimacode/aima-python/search.py and made the required modifications for the 2d problem

#######################################################################################################

################################## HOMEWORK 2 : CS 44444 ##############################################

#######################################################################################################

import numpy as np;
import math;
#------ source : github.com/aimacode/aima-python/search.py
class Queue:

    def __init__(self):
        raise NotImplementedError

    def extend(self, items):
        for item in items:
            self.append(item)

class PriorityQueue(Queue):

    def __init__(self, order=min, f=lambda x: x):
        self.A = []
        self.order = order
        self.f = f

    def append(self, item):
        bisect.insort(self.A, (self.f(item), item))

    def __len__(self):
        return len(self.A)

    def pop(self):
        if self.order == min:
            return self.A.pop(0)[1]
        else:
            return self.A.pop()[1]

    def __contains__(self, item):
        return any(item == pair[1] for pair in self.A)

    def __getitem__(self, key):
        for _, item in self.A:
            if item == key:
                return item

    def __delitem__(self, key):
        for i, (value, item) in enumerate(self.A):
            if item == key:
                self.A.pop(i)

class Node:
    def __init__(self, location, parent=None, activity=None, route_cost=0):
        "Create a search tree Node, derived from a parent by an activity."
        self.location = location
        self.parent = parent
        self.activity = activity
        self.route_cost = route_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node %s>" % (self.location,)

    def __lt__(self, node):
        return self.location < node.location

    def expand(self, problem):
        "List the nodes reachable in one step from this node."
        return [self.child_node(problem, activity)
                for activity in problem.activitys(self.location)]

    def child_node(self, problem, activity):
        "[Figure 3.10]"
        next = problem.result(self.location, activity)
        return Node(next, self, activity,
                    problem.route_cost(self.route_cost, self.location,
                                      activity, next))

    def solution(self):
        "Return the sequence of activitys to go from the root to this node."
        return [node.activity for node in self.route()[1:]]

    def route(self):
        "Return a list of nodes forming the route from the root to this node."
        node, route_back = self, []
        while node:
            route_back.append(node)
            node = node.parent
        return list(reversed(route_back))

    def __eq__(self, other):
        return isinstance(other, Node) and self.location == other.location

    def __hash__(self):
        return hash(self.location)

import bisect

def is_in(elt, seq):
    """Similar to (elt in seq), but compares with 'is', not '=='."""
    return any(x is elt for x in seq)

def memoize(fn, slot=None):
    """Memoize fn: make it remember the computed value for any argument list.
    If slot is specified, store result in that slot of first argument.
    If slot is false, store results in a dictionary."""
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        def memoized_fn(*args):
            if args not in memoized_fn.cache:
                memoized_fn.cache[args] = fn(*args)
            return memoized_fn.cache[args]

        memoized_fn.cache = {}

    return memoized_fn


def best_first_graph_search(problem, f):

    f = memoize(f, 'f')
    node = Node(problem.initial)
    if problem.goal_test(node.location):
        return node
    frontier = PriorityQueue(min, f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.location):
            return node
        explored.add(node.location)
        for child in node.expand(problem):
            if child.location not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
    return None

def astar_search(problem, h):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.route_cost + h(n))

class Problem(object):

    """The abstract class for a formal problem.  You should subclass
    this and implement the methods activitys and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial location, and possibly a goal
        location, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def activitys(self, location):
        """Return the activitys that can be executed in the given
        location. The result would typically be a list, but if there are
        many activitys, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, location, activity):
        """Return the location that results from executing the given
        activity in the given location. The activity must be one of
        self.activitys(location)."""
        raise NotImplementedError

    def goal_test(self, location):
        """Return True if the location is a goal. The default method compares the
        location to self.goal or checks for location in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(location, self.goal)
        else:
            return location == self.goal

    def route_cost(self, c, location1, activity, location2):
        """Return the cost of a solution route that arrives at location2 from
        location1 via activity, assuming cost c to get up to location1. If the problem
        is such that the route doesn't matter, this function will only look at
        location2.  If the route does matter, it will consider c and maybe location1
        and activity. The default method costs 1 for every step in the route."""
        return c + 1

    def value(self, location):
        """For optimization problems, each location has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError

########################################### Modification for Homework ######################
class VacuumWorld(Problem):
   
    def activitys(self, location):
        activitys = ["Suck"]
        # If we're not in square 1, 2 or 3
        if (location[9] > 3):
            activitys += ["Move Up"]
        if (location[9] % 3 != 1):
            activitys += ["Move Left"]
        if (location[9] % 3 != 0):
            activitys += ["Move Right"]
        if (location[9] < 7):
            activitys += ["Move Down"]
        return activitys
    
    def goal_test(self, location):
        # if the Goal is reached Stop the Program
        sublist = location[:9]
        return sublist == ("Clean", "Clean", "Clean", "Clean", "Clean", "Clean", "Clean", "Clean", "Clean")
    
    def result(self, location, activity):
        # change location (a tuple) to a list to make it mutable
        mutable_location = list(location)
        if activity == "Suck":
            mutable_location[mutable_location[9]-1] = "Clean"
        elif activity == "Move Up":
            mutable_location[9] -= 3
        elif activity == "Move Down":
            mutable_location[9] += 3
        elif activity == "Move Left":
            mutable_location[9] -= 1
        elif activity == "Move Right":
            mutable_location[9] += 1
       
        return tuple(mutable_location)
    
    def route_cost(self, c, location1, activity, location2):
        cost = 1
        # Each Action Cost =  1 point / Penalty = for Dirty = 2 points
        for element in location2[:9]:
            if element=="Dirty":
                cost+=2
        return cost

# create the problem
init_location = ("Dirty", "Dirty", "Dirty", "Clean", "Clean", "Clean", "Clean", "Clean", "Clean", 5)

problem = VacuumWorld(init_location)

def shortest_distance(n,p):
#### this function gives us the shortest distance to reach from current postion(1 t0 9) to any postion (p)
    if n==1:
        d_fast = [0,1,2,1,2,3,2,3,4]
        return d_fast[p]
    if n==2:
        d_fast = [1,0,1,2,1,2,3,2,3]
        return d_fast[p]
    if n==3:
        d_fast = [2,1,0,3,2,1,4,3,2]
        return d_fast[p]
    if n==4:
        d_fast = [1,2,3,0,1,2,1,2,3]
        return d_fast[p]
    if n==5:
        d_fast = [2,1,2,1,0,1,2,1,2]
        return d_fast[p]
    if n==6:
        d_fast = [3,2,1,2,1,0,3,2,1]
        return d_fast[p]
    if n==7:
        d_fast = [2,3,4,1,2,3,0,1,2]
        return d_fast[p]
    if n==8:
        d_fast = [3,2,3,2,1,2,1,0,1]
        return d_fast[p]
    if n==9:
        d_fast = [4,3,2,3,2,1,2,1,0]
        return d_fast[p]

def num_to_closest_dirty(location):
    d_fast = []
    for i in range(len(location[:9])):
        if location[i] == "Dirty":
            d_fast += [shortest_distance(location[9], i)]
    if not d_fast:
        return 0
    return min(d_fast)
    
# create heuristic function
def heuristic1(node):
    num_dirty = 0
    for element in node.location[:9]:
        if element=="Dirty" :
            num_dirty += 1
    dist_to_closest_dirty = num_to_closest_dirty(node.location)
    return num_dirty+dist_to_closest_dirty
   
    
# calling the A* Algorithm 
result = astar_search(problem, heuristic1)
route = []
node = result
while node!=None:
    route+=[node]
    node = node.parent

print("******* Initial Grid Configurations, States (Clean/Dirty) for the 3x3 Square Pool ******* ")

#A = np.array([[1, 2, 3], [4, 5, 6] , [7, 8 , 9]])
A = np.array([['1-Dirty', '2-Dirty    ', '3-Dirty'], ['4-Clean', '5-Clean/Agnt', '6-Clean'] , ['7-Clean', '8-Clean     ' , '9-Clean']])
print(A)
print("")
print("******Status Of Squares(1-9/ Clean or Dirty), position of Agent(1-9)*****")
route.reverse()
for i in route:
    print (i.location) 
####

w = 3*2; ### Acoounting for the initial cost (panalty for 3 dirty squares) when the program starts ##### 
routeCost = w
t = [];
n=0;
for i in route:
    routeCost += i.route_cost; 
    t.append(routeCost);
    n = n+1;
####

print("################ PART- A  ###################")


print("******Optimal route for A* search ******")
s=1
for j in route[1:]:
    f = t[s-1]
    print("Step #",s,":",j.activity, "            Previous Cost:", f)
    s=s+1;
print("Goal State Reached")

print("******Total Cost for the Operation *****")
print ("Total Cost : ",routeCost, "Points") 

print("################ PART- B  ###################")
      
def heuristic2(node):
    num_dirty = 0
    for element in node.location[:9]:
        if element=="Dirty" :
            num_dirty += 1
    dist_to_closest_dirty = 1
    return 1     
         
result = astar_search(problem, heuristic2)
route = []
node = result
while node!=None:
    route+=[node]
    node = node.parent

print("******* Initial Grid Configurations, States (Clean/Dirty) for the 3x3 Square Pool ******* ")

#A = np.array([[1, 2, 3], [4, 5, 6] , [7, 8 , 9]])
A = np.array([['1-Dirty', '2-Dirty    ', '3-Dirty'], ['4-Clean', '5-Clean/Agnt', '6-Clean'] , ['7-Clean', '8-Clean     ' , '9-Clean']])
print(A)
print("")
####

w = 3*2; ### Acoounting for the initial cost (panalty for 3 dirty squares) when the program starts ##### 
routeCost = w
t = [];
n=0;
for i in route:
    routeCost += i.route_cost; 
    t.append(routeCost);
    n = n+1;
routeCost = routeCost + w;
####


print("******Optimal route for A* search ******")
s=1
for j in route[1:]:
    f = t[s-1]
    print("Step #",s,":",j.activity,)
    s=s+1;
print("Goal State Reached")

print("******Total Cost for the Operation *****")
print ("Total Cost : ",routeCost, "Points")      
      