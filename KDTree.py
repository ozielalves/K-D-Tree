# -*- coding utf-8 -*-

from __future__ import print_function

""" Methods do implement:
	Node's:
		[x] __init__()
		[x] (@)is_leaf( ) - If a Node has no subnodes
		[x] preorder( ) - root, left, right
		[x] inorder( ) - left, root, right
		[x] postorder( ) - left, rigth, root
		[x] (@)children( ) - Iterator for the non-empty child of the Node
		[x] set_child(index, child) - Sets one of the Node's child, 0-left 1-right
		[x] height( ) - Subtree Height, not considering empty leaf-nodes
		[x] get_child_pos(child) - returns the position of the given child
		[x] __repr__( ) - An node representation
		[x] __nonzero__( ) - Returns if the node's data is not none
		[x] __eq__(other) - Returns if another node is iqual to self
		[x] __hash__ - Hashes the Node
		[x] require_axis(function) - Decorator to check if the function's object has
									 axis and sel_axis members
	KDNode(Node):
		[x] __init__ ( ) - The plus: Dimensions
		[x] (*)add(point) - Adds a point to the current node or decends to one of its 
					   		children (iteratively)
		[x] (*)create_subnode(data) - Subnode to the current node
		[x] (*)find_replacement( ) - Just find a replacement to the current node Returns
									 a tuple(replacement-node, replacement-dad-node)
		[x] should_remove(point, node) - Checks if the self's point matches
		[x] (*)remove(point, node=None) - Removes the node with the given point
		[x] (*)_remove(point) - Removes too (find out why)
		[x] (@)is_balanced( ) - True if the (sub)Tree is balanced, False otherwise
		[x] rebalance( ) - Returns the possibly new root of the rebalanced tree
		[x] axis_dist(point, axis) - Squared distance at the given axis between the 
									 current and an given node
		[x] dist(point) - Squared distance between the current node and the given
						  point
		[x] search_knn(point, k, dist=None) - Return the K nearest neighbors of the 
											  point and their distances (not a node)
		[x] _search_node(point, k, results, get_dist, counter) - Needed on Knn 
		[x] (*)search_nn(point, dist=None) - Search the nearest node of the given point
		[x] _search_nn_dist(point, dist, results, get_dist)
		[x] (*)search_nn_dist(point, distance, best=None) - Search nearest Node given
															point within given dist
		[x] (*)is_valid( ) - If each node splits correctly
		[x] extreme_child(sel_func, axis) - Returns a child of the subtree (min/max)
	
	[x] create(...) - Creats a KD Tree from a list of points
	[x] check_dimensionality(point_list, dimensions=None)
	[x] level_order(tree, include_all=False) - Iterator over the tree in level-order
	[x] visualize(...) - Prints the tree to stdout
	
	Legend: 
		(*) - Means that the function requieres axis
		(@) - Property """

"""A Python implemntation of a K-d Tree
https://en.wikipedia.org/wiki/K-d_tree
"""

import heapq
import itertools
import operator
import math
from collections import deque
from functools import wraps

class Node(object):
    # Each node represents its subtree

    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right


    @property
    def is_leaf(self):
        # If is empty or if the node has no subnodes
        
        return (not self.data) or \
               (all(not bool(c) for c, p in self.children))


    def preorder(self):
        # Iterator for nodes: root, left, right

        if not self:
            return

        yield self

        if self.left:
            for x in self.left.preorder():
                yield x

        if self.right:
            for x in self.right.preorder():
                yield x


    def inorder(self):
        # Iterator for nodes: left, root, right
        # Used on rebalance

        if not self:
            return

        if self.left:
            for x in self.left.inorder():
                yield x

        yield self

        if self.right:
            for x in self.right.inorder():
                yield x


    def postorder(self):
        # Iterator for nodes: right, left, root

        if not self:
            return

        if self.left:
            for x in self.left.postorder():
                yield x

        if self.right:
            for x in self.right.postorder():
                yield x

        yield self


    @property
    def children(self):
        # An iterator for the non-empty Node's child as:
        # Tuple(Node, pos), pos 0: left subnode; pos 1: right subnode 

        if self.left and self.left.data is not None:
            yield self.left, 0
        if self.right and self.right.data is not None:
            yield self.right, 1


    def set_child(self, index, child):
        # Sets one of the Node's child:
        # 0: Left; 1: Right 
        # Each child has an id that refers to its position
        # Used on the _remove from Tree

        if index == 0:
            self.left = child
        else:
            self.right = child


    def height(self):
        # Height of the (sub)Tree, no empty leaf-nodes considering 
        # Used on balance check

        min_height = int(bool(self))
        return max([min_height] + [c.height()+1 for c, p in self.children])


    def get_child_pos(self, child):
        # Returns the position (0 or 1) of the given child, considering
        # 0 for left, 1 for right and None otherwise 
        # Used on _remove from Tree

        for c, pos in self.children:
            if child == c:
                return pos


    def __repr__(self):
        # A simple representation method for the Node 
        return '<%(cls)s - %(data)s>' % \
            dict(cls=self.__class__.__name__, data=repr(self.data))


    def __nonzero__(self):
        #  Just returns if the node(data) is not None 
        return self.data is not None

    __bool__ = __nonzero__

    def __eq__(self, other):
        #  Just returns if the Node is equal to another
        if isinstance(other, tuple):
            return self.data == other
        else:
            return self.data == other.data

    def __hash__(self):
        return id(self)


def require_axis(f):
    # Checks if the object of the function has axis and sel_axis members

    @wraps(f)
    def _wrapper(self, *args, **kwargs):
        if None in (self.axis, self.sel_axis):
            raise ValueError('%(func_name) requires the node %(node)s '
                    'to have an axis and a sel_axis function' %
                    dict(func_name=f.__name__, node=repr(self)))

        return f(self, *args, **kwargs)

    return _wrapper



class KDNode(Node):
    # Node containing K-D Tree specific data and methods 
    # sel_axis is needed when creating subnodes, receives the axis of the parent
    # and returns the axis of the child node


    def __init__(self, data=None, left=None, right=None, axis=None,
            sel_axis=None, dimensions=None):

        super(KDNode, self).__init__(data, left, right)
        self.axis = axis
        self.sel_axis = sel_axis
        self.dimensions = dimensions


    @require_axis
    def add(self, point):
        # Adds an point to the current node or iteratively decends to one of its
        # children (Only callable to the topmost tree.) - the root

        current = self
        while True:
            check_dimensionality([point], dimensions=current.dimensions)

            # Adding has hit an empty leaf-node, add here
            if current.data is None:
                current.data = point
                return current

            # split on self.axis, recurse either left or right
            if point[current.axis] < current.data[current.axis]:
                if current.left is None:
                    current.left = current.create_subnode(point)
                    return current.left
                else:
                    current = current.left
            else:
                if current.right is None:
                    current.right = current.create_subnode(point)
                    return current.right
                else:
                    current = current.right


    @require_axis
    def create_subnode(self, data):
        # Creates a subnode for an given node.

        return self.__class__(data,
                axis=self.sel_axis(self.axis),
                sel_axis=self.sel_axis,
                dimensions=self.dimensions)


    @require_axis
    def find_replacement(self):
        # Tool for the rebalance, finds a replacement for the current node, returned as:
        # tuple(replacement_node, replacement-dad-node)

        if self.right:
            child, parent = self.right.extreme_child(min, self.axis)
        else:
            child, parent = self.left.extreme_child(max, self.axis)

        return (child, parent if parent is not None else self)


    def should_remove(self, point, node):
        # Tool for the remove. Checks if selfs point (maybe id) matches
        if not self.data == point:
            return False

        return (node is None) or (node is self)


    @require_axis
    def remove(self, point, node=None):
        # A tool for the real remove
        # Remove the given point's node from the tree
        
        # The new root node os the (sub)Tree is returned
        # If mutiple points are matching only one is removed

        # The node parameter if used for checking the ID(once the removal candidate
        # is decided)

        # Recursion has reached an empty leaf node, nothing here to delete
        if not self:
            return

        # Recursion has reached the node to be deleted
        if self.should_remove(point, node):
            return self._remove(point)

        # Remove direct subnode
        if self.left and self.left.should_remove(point, node):
            self.left = self.left._remove(point)

        elif self.right and self.right.should_remove(point, node):
            self.right = self.right._remove(point)

        # Recurse to subtrees
        if point[self.axis] <= self.data[self.axis]:
            if self.left:
                self.left = self.left.remove(point, node)

        if point[self.axis] >= self.data[self.axis]:
            if self.right:
                self.right = self.right.remove(point, node)

        return self


    @require_axis
    def _remove(self, point):
        # we have reached the node to be deleted here

        # deleting a leaf node is trivial
        if self.is_leaf:
            self.data = None
            return self

        # we have to delete a non-leaf node here

        # find a replacement for the node (will be the new subtree-root)
        root, max_p = self.find_replacement()

        # self and root swap positions
        tmp_l, tmp_r = self.left, self.right
        self.left, self.right = root.left, root.right
        root.left, root.right = tmp_l if tmp_l is not root else self, tmp_r if tmp_r is not root else self
        self.axis, root.axis = root.axis, self.axis

        # Special-case if we have not chosen a direct child as the replacement
        if max_p is not self:
            pos = max_p.get_child_pos(root)
            max_p.set_child(pos, self)
            max_p.remove(point, self)

        else:
            root.remove(point, self)

        return root


    @property
    def is_balanced(self):
        # True if the (sub)Tree is balanced
        # Tree is balanced if the heights of both subtrees differ at most by 1

        left_height = self.left.height() if self.left else 0
        right_height = self.right.height() if self.right else 0

        if abs(left_height - right_height) > 1:
            return False

        return all(c.is_balanced for c, _ in self.children)


    def rebalance(self):
        # Return the (possibly new) root of the rebalanced tree 

        return create([x.data for x in self.inorder()])


    def axis_dist(self, point, axis):
        # Squared distence at the given axis between the current Node
        # and the given point 
        return math.pow(self.data[axis] - point[axis], 2)


    def dist(self, point):
        # Squared distance between th current Node and the given point
        
        r = range(self.dimensions)
        return sum([self.axis_dist(point, i) for i in r])


    def search_knn(self, point, k, dist=None):
        # Return the k nearest neighbor of the current point and their distnaces
        # Point must be a point, not a node. 

        # k = the number of results to return. The founded results can be less
        # in case of equal distances. - Makes it flexible

        # dist = An distance function expecing two points and return a distance value.
        # Distance values can be any comparable type.

        # The result is an ordered list of tuples(node, distance) 

        # Dealing with user calling errors

        if k < 1:
            raise ValueError("k must be greater than 0.")

        if dist is None:
            get_dist = lambda n: n.dist(point)
        else:
            get_dist = lambda n: dist(n.data, point)

        results = []

        self._search_node(point, k, results, get_dist, itertools.count())

        # We sort the final result by the distance in the tuple
        # (<KdNode>, distance).
        return [(node, -d) for d, _, node in sorted(results, reverse=True)]


    def _search_node(self, point, k, results, get_dist, counter):
        if not self:
            return

        nodeDist = get_dist(self)

        # Add current node to the priority queue if it closer than
        # at least one point in the queue.
        #
        # If the heap is at its capacity, we need to check if the
        # current node is closer than the current farthest node, and if
        # so, replace it.
        item = (-nodeDist, next(counter), self)
        if len(results) >= k:
            if -nodeDist > results[0][0]:
                heapq.heapreplace(results, item)
        else:
            heapq.heappush(results, item)
        # get the splitting plane
        split_plane = self.data[self.axis]
        # get the squared distance between the point and the splitting plane
        # (squared since all distances are squared).
        plane_dist = point[self.axis] - split_plane
        plane_dist2 = plane_dist * plane_dist

        # Search the side of the splitting plane that the point is in
        if point[self.axis] < split_plane:
            if self.left is not None:
                self.left._search_node(point, k, results, get_dist, counter)
        else:
            if self.right is not None:
                self.right._search_node(point, k, results, get_dist, counter)

        # Search the other side of the splitting plane if it may contain
        # points closer than the farthest point in the current results.
        if -plane_dist2 > results[0][0] or len(results) < k:
            if point[self.axis] < self.data[self.axis]:
                if self.right is not None:
                    self.right._search_node(point, k, results, get_dist,
                                            counter)
            else:
                if self.left is not None:
                    self.left._search_node(point, k, results, get_dist,
                                           counter)


    @require_axis
    def search_nn(self, point, dist=None):
        # Return the nearest node of the given point
        # If an node is used as an point, the current node will be returned, not
        # its neighbor 

        # dist still a distance function, expecting two points and returning a 
        # distance value. Distance values can be any comparable type

        # The result is a tuple(node, distance)


        return next(iter(self.search_knn(point, 1, dist)), None)


    def _search_nn_dist(self, point, dist, results, get_dist):
        if not self:
            return

        nodeDist = get_dist(self)

        if nodeDist < dist:
            results.append(self.data)

        # get the splitting plane
        split_plane = self.data[self.axis]

        # Search the side of the splitting plane that the point is in
        if point[self.axis] <= split_plane + dist:
            if self.left is not None:
                self.left._search_nn_dist(point, dist, results, get_dist)
        if point[self.axis] >= split_plane - dist:
            if self.right is not None:
                self.right._search_nn_dist(point, dist, results, get_dist)


    @require_axis
    def search_nn_dist(self, point, distance, best=None):
        # Search the nearest node of the given point within the given distance
        
        results = []
        get_dist = lambda n: n.dist(point)

        self._search_nn_dist(point, distance, results, get_dist)
        return results


    @require_axis
    def is_valid(self):
        # Verifies recursively if the tree is valid 
        # If each node splits correctly, True; Otherwise, False

        if not self:
            return True

        if self.left and self.data[self.axis] < self.left.data[self.axis]:
            return False

        if self.right and self.data[self.axis] > self.right.data[self.axis]:
            return False

        return all(c.is_valid() for c, _ in self.children) or self.is_leaf


    def extreme_child(self, sel_func, axis):
        # Returns a child of the subTree and its parent
        # The child is selected by sel_func which is either min or max

        max_key = lambda child_parent: child_parent[0].data[axis]


        # we don't know our parent, so we include None
        me = [(self, None)] if self else []

        child_max = [c.extreme_child(sel_func, axis) for c, _ in self.children]
        # insert self for unknown parents
        child_max = [(c, p if p is not None else self) for c, p in child_max]

        candidates =  me + child_max

        if not candidates:
            return None, None

        return sel_func(candidates, key=max_key)



def create(point_list=None, dimensions=None, axis=0, sel_axis=None):
    # Creates a K-D Tree from a list of point (Same dimensionality)
    # If no points are given the number of dimensions has to be given insted
    # If both points and dimensions are given, the number must match
    # Axis is the axis on witch the root-node should split
    # Sel Axis(AXIS) is used when creating subnodes; recives the axis of the parent
    # and returns the axis of the child node.

    if not point_list and not dimensions:
        raise ValueError('either point_list or dimensions must be provided')

    elif point_list:
        dimensions = check_dimensionality(point_list, dimensions)

    # by default cycle through the axis
    sel_axis = sel_axis or (lambda prev_axis: (prev_axis+1) % dimensions)

    if not point_list:
        return KDNode(sel_axis=sel_axis, axis=axis, dimensions=dimensions)

    # Sort point list and choose median as pivot element
    point_list = list(point_list)
    point_list.sort(key=lambda point: point[axis])
    median = len(point_list) // 2

    loc   = point_list[median]
    left  = create(point_list[:median], dimensions, sel_axis(axis))
    right = create(point_list[median + 1:], dimensions, sel_axis(axis))
    return KDNode(loc, left, right, axis=axis, sel_axis=sel_axis, dimensions=dimensions)


def check_dimensionality(point_list, dimensions=None):
    dimensions = dimensions or len(point_list[0])
    for p in point_list:
        if len(p) != dimensions:
            raise ValueError('All Points in the point_list must have the same dimensionality')

    return dimensions



def level_order(tree, include_all=False):
    # Returns an iterator over the Tree in level-order 
    # If include_all is settet as True, empty parts of the tree are filled and
    # the iterator becmoes infinite (bcs of the dummy entries

    q = deque()
    q.append(tree)
    while q:
        node = q.popleft()
        yield node

        if include_all or node.left:
            q.append(node.left or node.__class__())

        if include_all or node.right:
            q.append(node.right or node.__class__())



def visualize(tree, max_level=100, node_width=10, left_padding=5):
    # Prints the tree on terminal

    height = min(max_level, tree.height()-1)
    max_width = pow(2, height)

    per_level = 1
    in_level  = 0
    level     = 0

    for node in level_order(tree, include_all=True):

        if in_level == 0:
            print()
            print()
            print(' '*left_padding, end=' ')

        width = int(max_width*node_width/per_level)

        node_str = (str(node.data) if node else '').center(width)
        print(node_str, end=' ')

        in_level += 1

        if in_level == per_level:
            in_level   = 0
            per_level *= 2
            level     += 1

        if level > height:
            break

    print()
    print()