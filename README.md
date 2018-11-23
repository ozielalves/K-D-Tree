# K-D-Tree

A repository for implementing a K-D Tree based on Python K-d Tree from [Stephen Kögl](https://python-kdtree.readthedocs.io/en/latest/)

#### ___The Kd Tree___
Is a space-partitioning data structure for organizing points in a k-dimensional space. k-d trees are a useful data structure for several applications, such as searches involving a multidimensional search key. K-d trees are a special case of binary space partitioning trees.
#### ___How it works?___
The k-d tree is a binary tree in which every leaf node is a k-dimensional point. Every non-leaf node can be thought of as implicitly generating a splitting hyperplane that divides the space into two parts, known as half-spaces. Points to the left of this hyperplane are represented by the left subtree of that node and points to the right of the hyperplane are represented by the right subtree. The hyperplane direction is chosen in the following way: every node in the tree is associated with one of the k-dimensions, with the hyperplane perpendicular to that dimension's axis.So, for example, if for a particular split the "x" axis is chosen, all points in the subtree with a smaller "x" value than the node will appear in the left subtree and all points with larger "x" value will be in the right subtree. In such a case, the hyperplane would be set by the x-value of the point, and its normal would be the unit x-axis.

Ref.: https://wikipedia.org/wiki/K-d_tree

## Implemented methods:
- Node's:
  - [x] __init__( ) - Data, Left, Right
  -	[x] __is_leaf__( ) - If a Node has no subnodes
  -	[x] __preorder__( ) - root, left, right
  -	[x] __inorder__( ) - left, root, right
  -	[x] __postorder__( ) - left, rigth, root
  -	[x] __children__( ) - Iterator for the non-empty child of the Node
  -	[x] __set_child__(index, child) - Sets one of the Node's child, 0-left 1-right
  -	[x] __height__( ) - Subtree Height, not considering empty leaf-nodes
  -	[x] __get_child_pos__(child) - returns the position of the given child
  -	[x] __repr__( ) - An node representation
  -	[x] __nonzero__( ) - Returns if the node's data is not none
  -	[x] __eq__(other) - Returns if another node is iqual to self
  -	[x] __hash__( ) - Hashes the Node
  -	[x] __require_axis__(function) - Decorator to check if the function's object has
									 axis and sel_axis members
- KDNode(Node):
	-	[x] __init__( ) - The plus: Dimensions
	-	[x] __add__(point) - Adds a point to the current node or decends to one of its 
					   		children (iteratively)
	-	[x] __create_subnode__(data) - Subnode to the current node
	-	[x] __find_replacement__( ) - Just find a replacement to the current node Returns
									 a tuple(replacement-node, replacement-dad-node)
	-	[x] __should_remove__(point, node) - Checks if the self's point matches
	-	[x] __remove__(point, node=None) - Removes the node with the given point
	-	[x] __is_balanced__( ) - True if the (sub)Tree is balanced, False otherwise
	-	[x] __rebalance__( ) - Returns the possibly new root of the rebalanced tree
	-	[x] __axis_dist__(point, axis) - Squared distance at the given axis between the 
									 current and an given node
	-	[x] __dist__(point) - Squared distance between the current node and the given
						  point
	-	[x] __search_knn__(point, k, dist=None) - Return the K nearest neighbors of the 
											  point and their distances (not a node)
	-	[x] __search_node__(point, k, results, get_dist, counter) - Needed on Knn 
	-	[x] __search_nn__(point, dist=None) - Search the nearest node of the given point
	-	[x] __search_nn_dist__(point, dist, results, get_dist)
	-	[x] __search_nn_dist__(point, distance, best=None) - Search nearest Node given
															point within given dist
	-	[x] __is_valid__( ) - If each node splits correctly
	-	[x] __extreme_child__(sel_func, axis) - Returns a child of the subtree (min/max)
	
- [x] __create__(...) - Creats a KD Tree from a list of points
-	[x] __check_dimensionality__(point_list, dimensions=None)
-	[x] __level_order__(tree, include_all=False) - Iterator over the tree in level-order
-	[x] __visualize__(...) - Prints the tree to stdout
	
## How to compile

First of all, obviously this repository is needed to work as desired, so we must to download this repo. As we know that `git` is a tremendous tool for any programmer, then we must have git installed. Make sure also that you have installed python 3 version on your computer.

```bash
# Using 'git clone' to clone this repo into desired directory:
$ git clone https://github.com/ozielalves/K-D-Tree.git

# Enter repo:
$ cd K-D-Tree

# To compile the test cases inside of path's root:
$ python3 driver.py
```
## GitHub Repository:

*https://github.com/ozielalves/K-D-Tree*


## Authorship

Program developed by [_Oziel Alves_](https://github.com/ozielalves) (*ozielalves@ufrn.edu.br*), 2018.2

&copy; IMD/UFRN 2018.
