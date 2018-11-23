from __future__ import absolute_import

import sys
import random
import logging
import unittest
import doctest
import collections
from itertools import islice

# import after starting coverage, to ensure that import-time code is covered
# import Node
import KDTree

class RemoveTest(unittest.TestCase):

    def test_remove_duplicates(self):
        """ creates a tree with only duplicate points, and removes them all """

        points = [(1,1)] * 100
        tree = KDTree.create(points)
        self.assertTrue(tree.is_valid())

        random.shuffle(points)
        while points:
            point = points.pop(0)

            tree = tree.remove(point)

            # Check if the Tree is valid after the removal
            self.assertTrue(tree.is_valid())

            # Check if the removal reduced the number of nodes by 1 (not more, not less)
            remaining_points = len(points)
            nodes_in_tree = len(list(tree.inorder()))
            self.assertEqual(nodes_in_tree, remaining_points)

    def test_remove(self, num=100):
        """ Tests random removal from a tree, multiple times """

        for i in range(num):
            self.do_random_remove()

    def do_random_remove(self):
        """ Creates a random tree, removes all points in random order """

        points = list(set(islice(random_points(), 0, 20)))
        tree =  KDTree.create(points)
        self.assertTrue(tree.is_valid())

        random.shuffle(points)
        while points:
            point = points.pop(0)

            tree = tree.remove(point)

            # Check if the Tree is valid after the removal
            self.assertTrue(tree.is_valid())

            # Check if the point has actually been removed
            self.assertTrue(point not in [n.data for n in tree.inorder()])

            # Check if the removal reduced the number of nodes by 1 (not more, not less)
            remaining_points = len(points)
            nodes_in_tree = len(list(tree.inorder()))
            self.assertEqual(nodes_in_tree, remaining_points)

    def test_remove_empty_tree(self):
        tree = KDTree.create(dimensions=2)
        tree.remove( (1, 2) )
        self.assertFalse(bool(tree))

    print("\n>> Remove tests: Passed")


class AddTest(unittest.TestCase):

    def test_add(self, num=10):
        """ Tests random additions to a tree, multiple times """

        for i in range(num):
            self.do_random_add()

    def do_random_add(self, num_points=100):

        points = list(set(islice(random_points(), 0, num_points)))
        tree = KDTree.create(dimensions=len(points[0]))
        for n, point in enumerate(points, 1):

            tree.add(point)

            self.assertTrue(tree.is_valid())

            self.assertTrue(point in [node.data for node in tree.inorder()])

            nodes_in_tree = len(list(tree.inorder()))
            self.assertEqual(nodes_in_tree, n)

    print("\n>> Add tests: Passed") 

class InvalidTreeTests(unittest.TestCase):

    def test_invalid_child(self):
        """ Children on wrong subtree invalidate Tree """
        child = KDTree.KDNode( (3, 2) )
        child.axis = 2
        tree = KDTree.create([(2, 3)])
        tree.left=child
        self.assertFalse(tree.is_valid())

        tree = KDTree.create([(4, 1)])
        tree.right=child
        self.assertFalse(tree.is_valid())

    def test_different_dimensions(self):
        """ Can't create Tree for Points of different dimensions """
        points = [ (1, 2), (2, 3, 4) ]
        self.assertRaises(ValueError, KDTree.create, points)

    print("\n>> Invalid Tree tests: Passed")


class TreeTraversals(unittest.TestCase):

    def test_same_length(self):
        tree = random_tree()

        inorder_len = len(list(tree.inorder()))
        preorder_len = len(list(tree.preorder()))
        postorder_len = len(list(tree.postorder()))

        self.assertEqual(inorder_len, preorder_len)
        self.assertEqual(preorder_len, postorder_len)

    print("\n>> Tree traversal tests: Passed")



class BalanceTests(unittest.TestCase):

    def test_rebalance(self):

        tree = random_tree(1)
        while tree.is_balanced:
            tree.add(random_point())

        tree = tree.rebalance()
        self.assertTrue(tree.is_balanced)
        
    print("\n>> Balance tests: Passed")



class NearestNeighbor(unittest.TestCase):

    def test_search_knn(self):
        points = [(50, 20), (51, 19), (1, 80)]
        tree = KDTree.create(points)
        point = (48, 18)

        all_dist = []
        for p in tree.inorder():
            dist = p.dist(point)
            all_dist.append([p, dist])

        all_dist = sorted(all_dist, key = lambda n:n[1])

        result = tree.search_knn(point, 1)
        self.assertEqual(result[0][1], all_dist[0][1])

        result = tree.search_knn(point, 2)
        self.assertEqual(result[0][1], all_dist[0][1])
        self.assertEqual(result[1][1], all_dist[1][1])

        result = tree.search_knn(point, 3)
        self.assertEqual(result[0][1], all_dist[0][1])
        self.assertEqual(result[1][1], all_dist[1][1])
        self.assertEqual(result[2][1], all_dist[2][1])

    def test_search_nn(self, nodes=100):
        points = list(islice(random_points(), 0, nodes))
        tree = KDTree.create(points)
        point = random_point()

        nn, dist = tree.search_nn(point)
        best, best_dist = self.find_best(tree, point)
        self.assertEqual(best_dist, dist, msg=', '.join(repr(p) for p in points) + ' / ' + repr(point))

    def find_best(self, tree, point):
        best = None
        best_dist = None
        for p in tree.inorder():
            dist = p.dist(point)
            if best is None or dist < best_dist:
                best = p
                best_dist = dist
        return best, best_dist

    print("\n>> KNN search test: Passed")
    print("\n>> NN Search test: Passed")
    print("\n>> Find best test: Passed")

class PointTypeTests(unittest.TestCase):
    """ test using different types as points """

    def test_point_types(self):
        emptyTree = KDTree.create(dimensions=3)
        point1 = (2, 3, 4)
        point2 = [4, 5, 6]
        Point = collections.namedtuple('Point', 'x y z')
        point3 = Point(5, 3, 2)
        tree = KDTree.create([point1, point2, point3])
        res, dist = tree.search_nn( (1, 2, 3) )

        self.assertEqual(res, KDTree.KDNode( (2, 3, 4) ))
        
    print("\n>> Point type tests: Passed")


def random_tree(nodes=20, dimensions=3, minval=0, maxval=100):
    points = list(islice(random_points(), 0, nodes))
    tree = KDTree.create(points)
    return tree

def random_point(dimensions=3, minval=0, maxval=100):
    return tuple(random.randint(minval, maxval) for _ in range(dimensions))

def random_points(dimensions=3, minval=0, maxval=100):
    while True:
        yield random_point(dimensions, minval, maxval)


if __name__ == "__main__":
    unittest.main()