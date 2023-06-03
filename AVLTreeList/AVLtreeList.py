# username - complete info
# id1      - 209424282
# name1    - Amir Levy
# id2      - 212050835
# name2    - Elad Salama


"""A class representing a node in an AVL tree"""


class AVLNode(object):
    """Constructor

    @type value: str
    @param value: data of your node
    """

    def __init__(self, value, isReal=True):
        self.value = value
        self.left = None
        self.right = None
        self.parent = None
        self.height = 0 if isReal else -1
        self.size = 1 if isReal else 0

    """returns the left child
       
    @rtype: AVLNode
    * every leaf have a virtual child:  virtual.value = None
    @returns: the left child of self, None if there is no left child
    """

    def getLeft(self):
        return self.left

    """returns the right child

    @rtype: AVLNode
    * every leaf have a virtual child:  virtual.value = None
    @returns: the right child of self, None if there is no right child
    """

    def getRight(self):
        return self.right

    """returns the parent 

    @rtype: AVLNode
    @returns: the parent of self, None if there is no parent
    """

    def getParent(self):
        return self.parent

    """return the value
    @rtype: str
    @returns: the value of self, None if the node is virtual
    """

    def getValue(self):
        return self.value

    """return the balance factor
    
    @type node: AVLnode
    @rtype: int
    @returns: the bf of node, None if the node is virtual
    """

    def getBalanceFactor(self):
        return self.getLeft().getHeight() - self.getRight().getHeight()

    """returns the height

    @rtype: int
    @returns: the height of self, -1 if the node is virtual
    """

    def getHeight(self):
        return self.height

    """returns the size

    @rtype: int
    @returns: the size of self, 0 if the node is virtual
    """

    def getSize(self):
        return self.size

    """sets left child

    @type node: AVLNode
    @param node: a node
    """

    def setLeft(self, node):
        if self.isRealNode():
            self.left = node

    """sets right child

    @type node: AVLNode
    @param node: a node
    """

    def setRight(self, node):
        if self.isRealNode():
            self.right = node

    """sets parent

    @type node: AVLNode
    @param node: a node
    """

    def setParent(self, node):
        self.parent = node

    """sets value

    @type value: str
    @param value: data
    """

    def setValue(self, value):
        if self.isRealNode():
            self.value = value

    """sets the height of the node

    @type h: int
    @param h: the height
    """

    def setHeight(self, h):
        self.height = h

    """sets the size of the node

    @type s: int
    @param s: the size
    """

    def setSize(self, s):
        self.size = s

    """returns whether self is not a virtual node 

    @rtype: bool
    @returns: False if self is a virtual node, True otherwise.
    """

    def isRealNode(self):
        return self.height != -1

    """returns whether self is not a leaf  

    @rtype: bool
    @returns:True if self is leaf, false otherwise
    """

    def isLeaf(self):
        return not self.getLeft().isRealNode() and not self.getRight().isRealNode()

    def __repr__(self):
        return self.value


"""
A class implementing the ADT list, using an AVL tree.
"""


class AVLTreeList(object):

    def __init__(self):
        self.root = None
        self.max = None
        self.min = None
        self.len = 0

    """returns whether the list is empty

    @rtype: bool
    @returns: True if the list is empty, False otherwise
    
    complexity - O(1)
    """

    def empty(self):
        return self.len == 0

    """retrieves the value of the i'th item in the list

    @type i: int
    @pre: 0 <= i < self.length()
    @param i: index in the list
    @rtype: str
    @returns: the the value of the i'th item in the list
    
    complexity - O(log(n))
    """

    def retrieve(self, i):
        return self.select(i + 1).getValue()

    """ find the node which rank is i
    @pre : 0<i
    
    complexity - O(log(n))
    """

    def select(self, i):
        def selectRec(node, j):
            if node.getLeft().isRealNode():
                cur = node.getLeft().getSize() + 1
            else:
                cur = 1
            if j == cur:
                return node
            elif j < cur:
                return selectRec(node.getLeft(), j)
            elif j > cur:
                return selectRec(node.getRight(), j - cur)

        return selectRec(self.root, i)

    """inserts val at position i in the list

    @type i: int
    @pre: 0 <= i <= self.length()
    @param i: The intended index in the list to which we insert val
    @type val: str
    @param val: the value we inserts
    @rtype: list
    @returns: the number of rebalancing operation due to AVL rebalancing
    
    complexity - O(log(n))
    """

    def insert(self, i, val):
        node = self.insertRT(i, val)
        node = node.getParent()
        count = 0
        count = self.balanceHelp(node)
        return count, node

    """inserts val at position i in the list like regular rank Tree
    @type i: int
    @pre: 0 <= i <= self.length()
    @param i: The intended index in the list to which we insert val
    @type val: str
    @param val: the value we inserts
    @rtype: list
    @returns: the node we insert
    
    complexity - O(log(n))
    """

    def insertRT(self, i, val, isAVL=True):
        """ insert node with i, val into tree"""

        node_in = AVLNode(val)
        if self.len == 0:  # setting the root
            self.max = node_in
            self.min = node_in
            self.root = node_in
            self.len += 1
            self.virtualFix(node_in)
        elif i == self.len:  # setting the maximum
            self.createNewNode(node_in, self.select(self.length()), True)
            self.max = node_in
        elif i == 0:  # setting the minimum
            self.createNewNode(node_in, self.select(1), False)
            self.min = node_in
        else:  # setting some value
            node = self.select(i + 1)
            if node.getLeft().isRealNode() is False:
                self.createNewNode(node_in, node, False)
            else:
                node_pre = self.predecessor(node)
                self.createNewNode(node_in, node_pre, True)
        if not isAVL:
            self.fixHeights(node_in, 1)
        return node_in

    """ inserts a new node inplace of virtual node
    
    complexity - O(log(n))
    """

    def createNewNode(self, node, parent, isRight):
        self.len += 1
        self.virtualFix(node)
        node.setParent(parent)
        if isRight:
            parent.setRight(node)
        else:
            parent.setLeft(node)
        self.updateSizes(node)

    """update the size of the node all the way to the root
    
    complexity - O(log(n))
    """

    def updateSizes(self, node):
        while node is not None:
            size = 0
            size += node.getRight().getSize()
            size += node.getLeft().getSize()
            node.setSize(1 + size)
            node = node.getParent()

    """ fix the height of the given node
    @type node : AVLnode
    @param node : the first node who needs height fixing
    @type curHeight : int
    @param curHeight : the current height
    
    complexity - O(log(n))
    """

    def fixHeights(self, node, curHeight):
        if node is None:
            return
        if not node.isRealNode():
            return
        val_l = node.getLeft().getHeight()
        val_r = node.getRight().getHeight()
        node.setHeight(max(val_r, val_l) + 1)
        self.fixHeights(node.getParent(), 1 + curHeight)

    """deletes the i'th item in the list

    @type i: int
    @pre: 0 <= i < self.length()
    @param i: The intended index in the list to be deleted
    @rtype: int
    @returns: the number of rebalancing operation due to AVL rebalancing
    
    complexity - O(log(n))
    """

    def delete(self, i):
        self.len = max(self.len - 1, 0)
        if self.len == 0 and i == 0:
            self.root = None
            self.len = 0
            self.max = None
            self.min = None
            return 1
        nodeToDel = self.select(i + 1)
        # easily delete a leaf
        if nodeToDel.isLeaf():
            nodePar = nodeToDel.getParent()
            if nodeToDel == self.max:
                self.max = nodePar
            if nodeToDel == self.min:
                self.min = nodePar
            if nodeToDel == nodePar.getLeft():
                nodePar.setLeft(AVLNode(None, isReal=False))
            else:
                nodePar.setRight(AVLNode(None, isReal=False))
            self.updateSizes(nodePar)
            count = self.balanceHelp(nodePar, notConcat=False)
            return count
        node = nodeToDel
        if nodeToDel != self.max:
            node = self.successor(nodeToDel)
        nodeToDel.setValue(node.getValue())
        # to make sure we delete a leaf
        while not node.isLeaf() and node != self.max:
            nodeToDel = node
            node = self.successor(nodeToDel)
            nodeToDel.setValue(node.getValue())
        if node == self.max:
            self.max = nodeToDel
            while not node.isLeaf() and node != self.min:
                nodeToDel = node
                node = self.predecessor(nodeToDel)
                nodeToDel.setValue(node.getValue())
        nodePar = node.getParent()
        # practically delete the leaf
        if node == nodePar.getLeft():
            nodePar.setLeft(AVLNode(None, isReal=False))
        else:
            nodePar.setRight(AVLNode(None, isReal=False))
        self.updateSizes(nodePar)
        # balance operations
        count = self.balanceHelp(nodePar, notConcat=False)
        return count

    """ make sure each Real Node has two sons
    
    complexity - O(1)
    """

    def virtualFix(self, node):
        if node.getLeft() is None:
            node.setLeft(AVLNode(None, isReal=False))
        if node.getRight() is None:
            node.setRight(AVLNode(None, isReal=False))

    """ right rotation to balance the tree
    @type B: AVLnode
    @param B: the "criminal" node (bf >=2 )
    
    complexity - O(log(n))
    """

    def rightRotation(self, B):
        """
             D                      D
              \                      \
               B                      A
              /  \     --->          /  \
            A     Br                Al   B
           /  \                    /   /   \
          Al  Ar                  v   Ar     Br
        /
       v
        """
        if B.getParent() is None:
            A = B.getLeft()
            self.root = A
            Ar = A.getRight()
            B.setLeft(Ar)
            Ar.setParent(B)
            A.setRight(B)
            B.setParent(A)
            A.setParent(None)
            self.fixHeights(B, A.getHeight())
            self.updateSizes(B)
            return
        if B == B.getParent().getLeft():
            C = True
        else:
            C = False
        D = B.getParent()
        A = B.getLeft()
        A.setParent(D)
        Ar = A.getRight()
        A.setRight(B)
        B.setParent(A)
        B.setLeft(Ar)
        Ar.setParent(B)
        if C:
            D.setLeft(A)
        else:
            D.setRight(A)
        self.virtualFix(B)
        self.fixHeights(B, A.getHeight())
        self.updateSizes(B)

    """ left rotation to balance the tree
    @type B: AVLnode
    @param B: the "criminal" node (bf >=2 )
    
    complexity - O(log(n))
    """

    def leftRotation(self, B):
        """
              D                      D
               \                      \
                B                      A
               /  \     --->          /  \
             Bl     A                B   Ar
                   /  \             / \    \
                  Al  Ar           Bl  Al    v
                       \
                        v
         """
        if B == self.root or B.getParent() is None:
            A = B.getRight()
            self.root = A
            Al = A.getLeft()
            B.setRight(Al)
            Al.setParent(B)
            A.setLeft(B)
            B.setParent(A)
            A.setParent(None)
            self.fixHeights(B, A.getHeight())
            self.updateSizes(B)
            return
        if B == B.getParent().getLeft():
            C = True
        else:
            C = False
        D = B.getParent()
        A = B.getRight()
        A.setParent(D)
        Al = A.getLeft()
        A.setLeft(B)
        B.setParent(A)
        B.setRight(Al)
        Al.setParent(B)
        if C:
            D.setLeft(A)
        else:
            D.setRight(A)
        self.fixHeights(B, A.getHeight())
        self.updateSizes(B)

    """ left then right rotation to balance the tree
     @type B: AVLnode
     @param B: the "criminal" node (bf >=2 )
     
     complexity - O(log(n))
     """

    def leftThenRightRotation(self, C):
        """
              D                      D
               \                       \
                C                       B
               /  \     --->         /     \
             A     Cr              A        C
           /  \                 /   \      /  \
          Al  B               Al    Bl    Br   Cr
            /  \                      \
          Bl    Br                     v
         /
        v
        """
        A = C.getLeft()
        self.leftRotation(A)
        self.rightRotation(C)

    """ right then left rotation to balance the tree
    @type B: AVLnode
    @param B: the "criminal" node (bf >=2 )
    
    complexity - O(log(n))
        """

    def rightThenLeftRotation(self, B):
        """
              D                      D
               \                       \
                B                       Al
               /  \     --->         /      \
             Bl    Br               B        Br
                  /  \            /   \     /  \
                 Al  Ar          Bl    Cl  Cr   Ar
                /  \                    \
               Cl   Cr                   v
              /
             v
        """
        Br = B.getRight()
        self.rightRotation(Br)
        self.leftRotation(B)

    """returns the value of the first item in the list


    @rtype: str
    @returns: the value of the first item, None if the list is empty
    
    complexity - O(1)
    """

    def first(self):
        return self.min.getValue()

    """returns the value of the last item in the list

    @rtype: str
    @returns: the value of the last item, None if the list is empty
    
    complexity - O(1)
    """

    def last(self):
        return self.max.getValue()

    """returns the  first item in the list


    @rtype: AVLnode
    @returns: t first item, None if the list is empty
    
    complexity - O(1)
    """

    def firstNode(self):
        return self.min

    """returns the last item in the list

    @rtype: AVLnode
    @returns: the last item, None if the list is empty
    
    complexity - O(1)
    """

    def lastNode(self):
        return self.max

    """returns an array representing list 

    @rtype: list
    @returns: a list of strings representing the data structure
    
    complexity - O(n)
    """

    def listToArray(self):
        arr = []
        if self.root is None:
            return arr

        def listToArrayRec(node, arr):
            if not node.isRealNode():
                return None
            if node.getLeft().isRealNode():
                listToArrayRec(node.getLeft(), arr)
            arr.append(node.getValue())
            if node.getRight().isRealNode():
                listToArrayRec(node.getRight(), arr)

        listToArrayRec(self.root, arr)
        return arr

    """returns the size of the list 

    @rtype: int
    @returns: the size of the list
    
    complexity - O(1)
    """

    def length(self):
        return self.len

    """splits the list at the i'th index

    @type i: int
    @pre: 0 <= i < self.length()
    @param i: The intended index in the list according to whom we split
    @rtype: list
    @returns: a list [left, val, right], where left is an AVLTreeList representing the list until index i-1,
    right is an AVLTreeList representing the list from index i+1, and val is the value at the i'th index.
    
    complexity - O(logn)
    """

    def split(self, i):
        #Q2 paramaters
        maxcost = 0
        cost = 0
        curCost = 0
        count = 0
        subPar = None

        x = self.select(i + 1)
        node = x
        parent = node.getParent()
        left = self.AVLfromNode(node.getLeft())
        right = self.AVLfromNode(node.getRight())
        while parent:
            if node == parent.getRight():
                subLeft = self.AVLfromNode(parent.getLeft())
                curCost = AVLTreeList.join(subLeft, AVLNode(parent.getValue()), left)  # Q2
                cost += curCost
                maxcost = max(maxcost, curCost)  # Q2
                count += 1  # Q2
                left = subLeft
            else:
                subRight = self.AVLfromNode(parent.getRight())
                curCost = AVLTreeList.join(right, AVLNode(parent.getValue()), subRight)  # Q2
                cost += curCost
                maxcost = max(maxcost, curCost)  # Q2
                count += 1  # Q2
            node = parent
            parent = node.getParent()
        if left.getRoot() is not None:
            left.setMax(left.getSubMax(left.getRoot()))
            left.setMin(left.getSubMin(left.getRoot()))
        if right.getRoot() is not None:
            right.setMax(right.getSubMax(right.getRoot()))
            right.setMin(right.getSubMin(right.getRoot()))
        print("avg cost: ", cost/count, " maxcost: ", maxcost) #Q2
        return [left, x.getValue(), right]

    """returns a sub AVLTreeList from self where the new root is node 

        @type node: AVLNode
        @pre: node is a real node in the tree
        @rtype: AVLTreeList
        @returns: a sub AVLTreeList from self where the new root is node

        complexity - O(1)
        """

    def AVLfromNode(self, node):
        if not node.isRealNode():
            return AVLTreeList()
        tree = AVLTreeList()
        node.setParent(None)
        tree.setRoot(node)
        tree.len = node.getSize()
        tree.max = None
        tree.min = None
        return tree

    """concatenates lst to self

    @type lst: AVLTreeList
    @param lst: a list to be concatenated after self
    @rtype: int
    @returns: the absolute value of the difference between the height of the AVL trees joined
    
    complexity - O(logn)
    """

    def concat(self, lst):
        x = AVLNode(None, isReal=False)
        return AVLTreeList.join(self, x, lst, isConcat=True)

    """joining 2 trees, one or two of them is empty
     @param T1: AVLTreeList . T1 < x (for every t in T1)
     @param x: AVLNode
     @param T2: AVLTreeList . T2 > x (for every t in T2)
     @post : all data is in T1, T2 is "gone"
     $ret : the absolute value of the difference between the height of the AVL trees joined
     complexity - O(logn)
     """

    @staticmethod
    def emptyJoin(T1, x, T2):
        if T1.getRoot() is None and T2.getRoot() is None:
            T1.setRoot(x)
            T1.setMax(x)
            T1.setMin(x)
            T1.setLen(1)
            return 0
        elif T1.getRoot() is None:
            val = T2.getRoot().getHeight() + 1
            T2.insert(0, x.getValue())
            T1.setRoot(T2.getRoot())
            T1.setMax(T2.lastNode())
            T1.setMin(T2.firstNode())
            T1.setLen(T2.length() + 1)
            return val
        elif T2.getRoot() is None:
            val = T1.getRoot().getHeight() + 1
            T1.insert(T1.length(), x.getValue())
            return val
        elif not T1.getRoot().isRealNode() and not T2.getRoot().isRealNode():
            T1.setRoot(x)
            T1.setMax(x)
            T1.setMin(x)
            T1.setLen(1)
            return 0
        elif not T1.getRoot().isRealNode():
            val = T2.getRoot().getHeight() + 1
            T2.insert(0, x.getValue())
            T1.setRoot(T2.getRoot())
            T1.setMax(T2.lastNode())
            T1.setMin(T2.firstNode())
            T1.setLen(T2.length() + 1)
            return val
        elif not T2.getRoot().isRealNode():
            val = T1.getRoot().getHeight() + 1
            T1.insert(T1.length(), x.getValue())
            return val

    """joining 2 trees
    @param T1: AVLTreeList . T1 < x (for every t in T1)
    @param x: AVLNode
    @param T2: AVLTreeList . T2 > x (for every t in T2)
    @post : all data is in T1, T2 is "gone"
    $ret : the absolute value of the difference between the height of the AVL trees joined
    complexity - O(logn)
    """

    @staticmethod
    def join(T1, x, T2, isConcat=False):
        if T1.getRoot() is None or T2.getRoot() is None or not \
                T1.getRoot().isRealNode() or not \
                T2.getRoot().isRealNode():
            return AVLTreeList.emptyJoin(T1, x, T2)
        rootChange = True
        val = abs(T1.getRoot().getHeight() - T2.getRoot().getHeight())
        tempLen = T2.length()
        if T1.getRoot().getHeight() >= T2.getRoot().getHeight():
            right = True
        else:
            right = False
        if isConcat:
            x = AVLNode(T1.last())
            T1.delete(T1.len - 1)
        if right:
            wantedHeight = T2.getRoot().getHeight()  # height = h
            node = T1.getRoot()
            # getting the first node which height is h or h-1
            while node.getHeight() != wantedHeight and node.getHeight() != wantedHeight - 1:
                if node.getRight().isRealNode():
                    node = node.getRight()
                else:
                    node = node.getLeft()
            # connect self as x.left and T2 as x.right
            x.setHeight(wantedHeight + 1)
            x.setSize(node.getSize() + T2.getRoot().getSize() + 1)
            x.setLeft(node)
            prevPar = node.getParent()
            node.setParent(x)
            x.setRight(T2.getRoot())
            T2.getRoot().setParent(x)
            if prevPar is not None:
                isRight = node == prevPar.getRight()
                x.setParent(prevPar)
                if isRight:
                    prevPar.setRight(x)
                else:
                    prevPar.setLeft(x)
            else:
                T1.setRoot(x)
            # start balancing from x
            count = 0
            T1.updateSizes(x.getParent())
            T1.max = T2.lastNode()
            T1.balanceHelp(x, notConcat=False)
        else:  # T2 is higher then self
            wantedHeight = T1.getRoot().getHeight()  # height = h
            node = T2.getRoot()
            # getting the first node which height is h or h-1
            while node.getHeight() != wantedHeight and node.getHeight() != wantedHeight - 1:
                if node.getLeft().isRealNode():
                    node = node.getLeft()
                else:
                    node = node.getRight()
            # connect T1 as x.left and T2 as x.right
            x.setHeight(wantedHeight + 1)
            x.setSize(node.getSize() + T1.getRoot().getSize() + 1)
            x.setLeft(T1.getRoot())
            prevPar = node.getParent()
            node.setParent(x)
            x.setRight(node)
            T1.getRoot().setParent(x)
            if prevPar is not None:
                isRight = node == prevPar.getRight()
                x.setParent(prevPar)
                if isRight:
                    prevPar.setRight(x)
                else:
                    prevPar.setLeft(x)
            else:
                T1.setRoot(x)
                rootChange = False
            # start balancing from x
            count = 0
            T1.updateSizes(x.getParent())
            T1.setMax(T2.lastNode())
            T1.balanceHelp(x, notConcat=False)
            if rootChange:
                T1.setRoot(T2.getRoot())
        T1.len += tempLen + 1
        return val

    """searches for a *value* in the list

    @type val: str
    @param val: a value to be searched
    @rtype: int
    @returns: the first index that contains val, -1 if not found.
    
    complexity - O(n)
    """

    def search(self, val):
        if self.root is None:
            return -1

        def searchRec(node, val):
            if node.getValue() == val:
                return node

            value = None
            if node.getLeft().isRealNode():
                value = searchRec(node.getLeft(), val)

            if value:
                return value

            if node.getRight().isRealNode():
                value = searchRec(node.getRight(), val)
            return value

        valNode = searchRec(self.root, val)
        if not valNode:
            return -1
        return self.getIndex(valNode)

    """returns the index of a tree's node in the array
        
        @pre: node is a real node in the tree
        @type node: AVLNode
        @rtype: int
        @returns: the index of a tree's node in the array

        complexity - O(logn)
        """

    def getIndex(self, node):
        index = node.getLeft().getSize()
        parent = node.getParent()

        while parent is not None:
            if parent.getRight() == node:
                index += parent.getLeft().getSize() + 1
            node = parent
            parent = node.getParent()

        return index

    """returns the root of the tree representing the list

    @rtype: AVLNode
    @returns: the root, None if the list is empty
    
    complexity - O(1)
    """

    def getRoot(self):
        return self.root

    """returns the minimum of the sub tree 

    @rtype: AVLNode
    @returns: the minimum of the sub tree, None if the list is empty
    
    complexity - O(log(n))
    """

    def getSubMin(self, node):
        cur = node
        while node is not None:
            if node.getLeft() is None:
                if node.isRealNode():
                    return node
                else:
                    break
            if not node.getLeft().isRealNode():
                break
            node = node.getLeft()
        return node

    """returns the maximum of the sub tree 

    @rtype: AVLNode
    @returns: the maximum of the sub tree, None if the list is empty
    
    complexity - O(log(n))
    """

    def getSubMax(self, node):
        cur = node
        while node is not None:
            if node.getRight() is None:
                if node.isRealNode():
                    return node
                else:
                    break
            if not node.getRight().isRealNode():
                break
            node = node.getRight()
        return node

    """returns the successor of the given node

    @rtype: AVLNode
    @returns: the successor if exist, None otherwise
    
    complexity - O(log(n))
    """

    def successor(self, node):
        if node == self.max:
            return node
        if node.getRight().isRealNode():
            return self.getSubMin(node.getRight())
        par = node.getParent()
        while par is not None:
            if node != par.getRight():
                break
            node = par
            par = par.getParent()
        return par

    """returns the predecessor of the given node
    @rtype: AVLNode
    @returns: the predecessor if exist, None otherwise
    
    complexity - O(log(n)) 
    """

    def predecessor(self, node):
        if node.getLeft() is not None:
            return self.getSubMax(node.getLeft())
        par = node.getParent()
        while par is not None:
            if node != par.getLeft():
                break
            node = par
            par = par.getParent()
        return par

    def setMin(self, node):
        self.min = node

    def setMax(self, node):
        self.max = node

    def setLen(self, length):
        self.len = length

    def setRoot(self, node):
        self.root = node

        """balancing the tree from node to root
        @rtype: int
        @returns: number of balancing operations

        complexity - O(log(n)) 
        """

    def balanceHelp(self, node, notConcat=True):
        count = 0
        while node is not None:
            change = False  # to avoid double calculating of balance operations (rotations or height change)
            cont = True
            # check if the height has changed
            curHeight = node.getHeight()
            newHeight = 1 + max(node.getLeft().getHeight(), node.getRight().getHeight())
            if curHeight == newHeight and notConcat:
                return count
            elif curHeight != newHeight and notConcat:
                change = True
                count += 1
                node.setHeight(newHeight)
            elif curHeight != newHeight and not notConcat:
                count += 1
                change = True
                node.setHeight(newHeight)

            # balancing operations

            curBalanceFactor = node.getBalanceFactor()
            if abs(curBalanceFactor) == 2:
                if curBalanceFactor == -2:
                    rightBf = node.getRight().getBalanceFactor()
                    if rightBf == 0:
                        self.leftRotation(node)
                        cont = False
                    elif rightBf == -1:  # L
                        self.leftRotation(node)
                        count += 1 - change
                    elif rightBf == 1:  # RL
                        self.rightThenLeftRotation(node)
                        count += 2 - change
                else:
                    leftBf = node.getLeft().getBalanceFactor()
                    if leftBf == 0:
                        self.rightRotation(node)
                        cont = False
                    elif leftBf == 1:  # R
                        self.rightRotation(node)
                        count += 1 - change
                    elif leftBf == -1:  # LR
                        self.leftThenRightRotation(node)
                        count += 2 - change

            if cont:
                node = node.getParent()
        return count

    ''' print the tree
    '''

    def __repr__(self):
        out = ""
        for row in printree(self.root):
            out = out + row + "\n"

        return out


def printree(t):
    """Print a textual representation of t
    bykey=True: show keys instead of values"""
    return trepr(t)


def trepr(t):
    """Return a list of textual representations of the levels in t
    bykey=True: show keys instead of values"""
    if t == None:
        return ["#"]

    thistr = str(t.getValue())

    return conc(trepr(t.left), thistr, trepr(t.right))


def conc(left, root, right):
    """Return a concatenation of textual represantations of
    a root node, its left node, and its right node
    root is a string, and left and right are lists of strings"""

    lwid = len(left[-1])
    rwid = len(right[-1])
    rootwid = len(root)

    result = [(lwid + 1) * " " + root + (rwid + 1) * " "]

    ls = leftspace(left[0])
    rs = rightspace(right[0])
    result.append(ls * " " + (lwid - ls) * "_" + "/" + rootwid * " " + "\\" + rs * "_" + (rwid - rs) * " ")

    for i in range(max(len(left), len(right))):
        row = ""
        if i < len(left):
            row += left[i]
        else:
            row += lwid * " "

        row += (rootwid + 2) * " "

        if i < len(right):
            row += right[i]
        else:
            row += rwid * " "

        result.append(row)

    return result


def leftspace(row):
    """helper for conc"""
    # row is the first row of a left node
    # returns the index of where the second whitespace starts
    i = len(row) - 1
    while row[i] == " ":
        i -= 1
    return i + 1


def rightspace(row):
    """helper for conc"""
    # row is the first row of a right node
    # returns the index of where the first whitespace ends
    i = 0
    while row[i] == " ":
        i += 1
    return i
