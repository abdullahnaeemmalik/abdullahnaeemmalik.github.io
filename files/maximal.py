import networkx as nx
import collections
import itertools
from itertools import count
import numpy as np
import matplotlib.pyplot as plt
import dgl
import torch

def PosetΔ(d):
    """I just found out that the same can be accomplished by np.arange(d+1)..
    Anyway, this function takes in an integer as input and produces the
    1D np array [0,1,...,d]
    """
    return  np.asarray([index for index in range(0, d+1)])

def SimplexΔ(n,d,divisibility=None):
    """
    This function develops the PosetΔ(d) by trying
    to find all ordered k-cells of [0,1,...,d] where 0≤k≤n+1. Takes as input
    two integers and produces a list of lists.
    """
    Xbulletn = [[v] for v in PosetΔ(d)]
    if divisibility == 'On':
        Xbulletn.pop(0)
        for i in range(n):
            Xbulletn = [r + [v] for r in Xbulletn for v in PosetΔ(d) if (v % r[-1] == 0 and v!=0)]
    if divisibility == None:
        for i in range(n):
            Xbulletn = [r + [v] for r in Xbulletn for v in PosetΔ(d) if r[-1] <= v]
    return Xbulletn

def DegeneracyName(n,j):
    return "s_{} {}".format(n,j)

def FaceName(n,j):
    return "d_{} {}".format(n,j)

def MappingPossibility(possiblesublist, possiblesuperlist):
    """
    The way the graph gets constructed is that it adds edges
    only between nodes if one node is a domain element of a face map or degeneracy map, and gets assigned 
    to an element of the range of its function. In consructing the graph, the two loops go over
    each possible node, and hence will compare many nodes that don't have a
    connection. For example, [0,1] cannot have a face value of [2]. This is
    decided simply by observing that [2] is not a sublist of [0,1].
    """
    c1 = collections.Counter(possiblesublist)
    #The counter method converts a list into a dictionary, where the keys are
    #the items in the list and values are the number of times each value appears
    c2 = collections.Counter(possiblesuperlist)
    for item, counted in c1.items():
        #The items method converts the dictionary into tuples
        #The double index in the iteration goes over each pair of values in
        #the tuple c1.items(). The count index is the value of the key-value pair.
        if counted > c2[item]:
            #The c2[item] returns the value of the key-value pair.
           return False
    return True

def IsNotDegenerate(temp):
    c = collections.Counter(temp)
    """Convert given list into dictionary with keys being list items 
    and values being the number of times these list items appear in 
    the list. 
    """
    for counted in c.values():
        #the values method for a dictionary simply returns the values of a key
        # as a list.
        if counted > 1:
            return False
    return True

def FirstDifferentIndex(list1,list2):
    for index, (first,second) in enumerate(zip(list1, list2)):
        """the enumreate method adds a counter, called index in this code
        to each tuple. The tuple created here is the vector made by 
        zipping together list1 and list2. The zip method stops
        after pairing the all entries of the smaller list with
        the bigger list
        """
        if first != second:
            return index
    return min(len(list2),len(list1))

def AddDegeneracyLabeltoNode(G,vertex):
    if not IsNotDegenerate(vertex):
        G.nodes["{}".format(vertex)]['degeneracy_status']='Degenerate'
    else:
        G.nodes["{}".format(vertex)]['degeneracy_status']='NotDegenerate'
        
#def AddConnectedComponentLabel(G,vertex):
#    Temp=list(nx.strongly_connected_components_recursive(G))
#    for index in range(0,len(Temp)):
#        if vertex in Temp[index]:
#            G.nodes["{}".format(vertex)]['connected_component']=index 
 
def AddHasNumberLabelToNode(G,vertex,number):
    temp = False
    for v in vertex:
        if v == number:
            temp = True
    if temp == True:
        G.nodes["{}".format(vertex)]["Vertex with {}".format(number)]='Yes'
    else:
        G.nodes["{}".format(vertex)]["Vertex with {}".format(number)]='No'
        
def MapValueAsGraph(d,order=None):
    """
    Here, we're taking input the number of 0-cells of the simplicial set and
    producing a mutli-directed graph, where each k-cell is in one layer of the graph and
    directed edges correspond to either degeneracy map or face map. 
    The graph build iteratively and builds two levels at a time, starting with
    the 0-cells. The commands here simply adds vertices to each appropriate
    level and connects them by an edge if there's a map between them. The 
    name of the map is added as an edge attribute. This is a dictionary
    associated with the edge. The level k of the k-cell is encoded in the vertices
    of this graph as node attributes. Other possible attributes here are whether
    or not the node is degenerate, whether or not the node has a prime number
    in it
    """
    Xgraph = nx.MultiDiGraph()
    if d == 0:
        Xgraph.add_edge('[0]', '[0,0]', func='s_0 0')
        return Xgraph
    for level in range(0,d):
        if order == 'divisible':
            x = SimplexΔ(level, d, divisibility = 'On')
            y = SimplexΔ(level+1, d, divisibility = 'On')
        if order == None:
            x = SimplexΔ(level,d)
            y = SimplexΔ(level+1,d)
        #if order == 'Binary':
            #for index in GenerateBinaryVS(d):
                #if HammingDistanceToZero(index) == level:
                    #x = index
                #if HammingDistanceToZero(index) == level +1:
                    #y = index
        for firstlistitem, secondlistitem in itertools.product(x,y):
            if MappingPossibility(firstlistitem, secondlistitem):
                if not IsNotDegenerate(secondlistitem):        
                    if len(set(secondlistitem)) == 1:
                        for firstindex in range(0,len(secondlistitem)):
                            Xgraph.add_edge(str(secondlistitem), str(firstlistitem), func=FaceName(firstindex,level))
                            Xgraph.nodes["{}".format(secondlistitem)]['level']=len(secondlistitem)
                            Xgraph.nodes["{}".format(firstlistitem)]['level']=len(firstlistitem)
                            #AddDegeneracyLabeltoNode(Xgraph,firstlistitem)
                            Xgraph.nodes["{}".format(secondlistitem)]['degeneracy_status']='Degenerate'
                        for secondindex in range(0,len(firstlistitem)):
                            Xgraph.add_edge(str(firstlistitem), str(secondlistitem), func=DegeneracyName(secondindex,level))
                            Xgraph.nodes["{}".format(secondlistitem)]['level']=len(secondlistitem)
                            Xgraph.nodes["{}".format(secondlistitem)]['degeneracy_status']='Degenerate'
                            Xgraph.nodes["{}".format(firstlistitem)]['level']=len(firstlistitem)
                            #AddDegeneracyLabeltoNode(Xgraph,firstlistitem)
                    else:
                        Xgraph.add_edge(str(secondlistitem), str(firstlistitem), func=FaceName(FirstDifferentIndex(firstlistitem,secondlistitem),level))
                        Xgraph.nodes["{}".format(secondlistitem)]['level']=len(secondlistitem)
                        Xgraph.nodes["{}".format(firstlistitem)]['level']=len(firstlistitem)
                        #AddDegeneracyLabeltoNode(Xgraph,firstlistitem)
                        #AddDegeneracyLabeltoNode(Xgraph,secondlistitem)
                        
                    if len(set(firstlistitem)) != 1:
                        if FirstDifferentIndex(firstlistitem,secondlistitem) != 0:
                            Xgraph.add_edge(str(firstlistitem), str(secondlistitem), func=DegeneracyName(FirstDifferentIndex(firstlistitem,secondlistitem)-1,level))
                            Xgraph.nodes["{}".format(secondlistitem)]['level']=len(secondlistitem)
                            Xgraph.nodes["{}".format(firstlistitem)]['level']=len(firstlistitem)
                            #AddDegeneracyLabeltoNode(Xgraph,firstlistitem)
                            #AddDegeneracyLabeltoNode(Xgraph,secondlistitem)
                else:
                    Xgraph.add_edge(str(secondlistitem), str(firstlistitem), func=FaceName(FirstDifferentIndex(firstlistitem,secondlistitem),level))
                    Xgraph.nodes["{}".format(firstlistitem)]['level']=len(firstlistitem)
                    Xgraph.nodes["{}".format(secondlistitem)]['level']=len(secondlistitem)
                    #AddDegeneracyLabeltoNode(Xgraph,firstlistitem)
                    #AddDegeneracyLabeltoNode(Xgraph,secondlistitem)
                for numbers in range(0,d+1):
                    AddHasNumberLabelToNode(Xgraph,firstlistitem,numbers)
                    AddHasNumberLabelToNode(Xgraph,secondlistitem,numbers)       
                    AddDegeneracyLabeltoNode(Xgraph,firstlistitem)
                    AddDegeneracyLabeltoNode(Xgraph,secondlistitem)
                    
    return Xgraph

def AddConnectedComponentLabel(G):
        Temp=list(nx.strongly_connected_components_recursive(G))
        for index in range(0,len(Temp)):
            for vertex in G.nodes:
                if vertex in Temp[index]:
                    G.nodes["{}".format(vertex)]['connected_component']=index 
        return G


def DrawSimplicialSet(G, node_attribute_to_remove=None, node_attribute=None,connected_components_colored=False):
    """
    This function takes in a graph G and assigns a 2D numpy array 
    with coordinates for each node. The x coodinates are in the top row 
    and y coordinates are in second row. After nodes are assigned their
    coordinates, then each node is paired with its position and 
    the output is then a dictionary of positions. This is returned 
    in the function.

    The nodes for G (called by G.nodes) is a list comprising of tuples. 
    The first entry is the node and the second entry of this tuple 
    is a dictionary comprising of the attributes of the node. 
    
    The approach is to first focus on the nodes, depending on level,
    and assign each node its position. To do this, first we need to 
    extract node label from G.nodes. 
    
    In this loop, we turn this whole list into a dictionary with the keys
    being the the level and the value being a list of all nodes.
    """
    layers = SortByNodeAttribute(G,'level')

    pos = None
    nodes = []

    height = len(layers)
    for i, layer in enumerate(layers.values()):
        width = len(layer)
        #This gives the total number of nodes in one layer
        xs = np.arange(0, width)
        #place each node in current layer at x=0, x=1, x=2, ..., x=width
        ys = np.repeat(i, width)
        #All y coordinates for each node stay the same, hence 'repeat'
        offset = ((width - 1)/2, (height - 1) / 2)
        #all nodes collectively have to be centered in each corresponding level
        #Otherwise, the graph we get looks like a right-angled triangle 
        #instead of the broom picture we have now
        layer_pos = np.column_stack([xs, ys]) - offset     
        if pos is None:
            pos = layer_pos
        else:
            pos = np.concatenate([pos, layer_pos])
        nodes.extend(layer)
    #We need to perform a final check here and see if any nodes 
    #overlap. If they do, all nodes in one layer are separated
    #by increasing their existing distance, depending on how 
    #big the node is. That's what the next line does.
    pos, pixel = SpreadNodesandGimmePixel(pos)
    #and we need to pair back nodes with position, because that's what nx.draw accepts as pos
    pos = dict(zip(nodes, pos))
    if node_attribute_to_remove != None:
        for n, data in list(G.nodes(data=True)):
            if data[node_attribute_to_remove] == node_attribute:
                del pos[n]
    plt.figure(figsize=(1.5*pixel,1.5*pixel))
    #This is where the pixels are used. Even if I do space the nodes far apart, using a default
    #pixels for the image generated still squishes the nodes together, because the
    #canvas isn't big enough.
    if connected_components_colored==True:
        G=AddConnectedComponentLabel(G)
        groups = set(nx.get_node_attributes(G,'connected_component').values())
        mapping = dict(zip(sorted(groups),count()))
        colors = [mapping[G.nodes[n]['connected_component']] for n in G.nodes()]
        nx.draw(G, pos, with_labels=True, node_size=[len(v) * height *100 for v in G.nodes()],connectionstyle='arc3, rad = 0.1', node_shape='o',node_color=colors)
    else :
        nx.draw(G, pos, with_labels=True, node_size=[len(v) * height *100 for v in G.nodes()],connectionstyle='arc3, rad = 0.1', node_shape='o')
    plt.savefig("plot.png")
    plt.show()
    print('Done with the drawing. Check your local project folder for image file')  
           
def SpreadNodesandGimmePixel(coordinates):
    """
    The graph produces what's needed, after accounting for the off-set. However,
    since the node labels tend to get bigger as we advance each level,
    the nodes start to overlap. This had to be corrected. The below code
    first checks if y-coordinates are the same (meaning on same level),
    and the first instance where two neighbor nodes are closer than 1 unit
    in the x-direction, then all the x coordinates in each level is spread 
    out by the maximum of all x-coordinates. This choice of 1 unit is 
    rather arbitrary. I don't know why this works. However, by adding
    the maximum of the x-coordinate to all x-coordinate, it pushes all 
    nodes apart. Using the maximum is probably an overkill..
    """
    for index in range(1,len(coordinates)):
        if coordinates[index,1] == coordinates[index+1,1]:
            if coordinates[index,0] - coordinates[index+1,0] < 1:
                coordinates[:,0] = coordinates[:,0] + max(abs(coordinates[:,0]))
                return coordinates, max(abs(coordinates[:,0]))

def SortByNodeAttribute(G,attribute):
    """
    Function takes in a graph and a string of a node attribute and produces
    an ordered dictionary with keys being the attribute and values being a list
    of all nodes which hold that attribute. Useful for sorting or checking
    whether assignment of a node is not crazy.
    I also think that one core reason why the built-in algorithms failed for
    the multidirected graph was because the resulting dictionary was assumed 
    to be ordered. There was a switch in the way dicionaries work in Python 
    starting from version 3.7. And because the position of the nodes relied
    on this absent order, the built-in methods in Networkx failed. Hence they
    had to be modified.
    """
    layers = {}
    for n, data in G.nodes(data=True):
        attributevalue = data[attribute]
        layers[attributevalue] = [n] + layers.get(attributevalue, [])
    return collections.OrderedDict(sorted(layers.items()))
    #This last command converts a dictionary into an ordered dictionary.

def RemoveNodebyAttribute(G, attribute, attributevalue):
    """
    This function takes in a graph and returns the same graph by 
    removing some nodes based on an attribute. So, if we want to 
    remove degenerate nodes from the graph made using 3 0-cells, 
    we'd use 
    RemoveNodebyAttribute(MapValue(2),'degeneracy_status', 'Degenerate')
    Since this function removes nodes from the original graph, and the
    drawing method accepts a graph, if we put this result in the argument for
    DrawSimplicialSet(), the position of the reduced node list is calculated afresh
     That means that it doesn't look like we're removing nodes from one existing graph.
    To accomplish this, I'm adding code in the draw function as an optional argument'
    """
    for n, data in list(G.nodes(data=True)):
        #I just learned that we can't modify a dictionary while it is being
        #iterated over because.. of course. The work around 
        #temp = dictionary
        #for index in dictionary
            #change (temp)
        #didn't seem to work, either. Apparently, setting temp = dictionary does not create
        #a copy of the dictionary. It only changes its alias, not the object itself.
        if data[attribute] == attributevalue:
            G.remove_node(n)
    return G

def MorphismFunction(SSet1,SSet2):
    nodedictionary1=dict(zip(range(len(SSet1.nodes)), SSet1.nodes))
    nodedictionary2=dict(zip(range(len(SSet2.nodes)), SSet2.nodes))
    print("First Simplicial Set has these numbered cells {}".format(nodedictionary1))
    print("First Simplicial Set has these numbered cells {}".format(nodedictionary2))
    NeedMap=0
    NeedMap=input("Any level functions to define? Input 0 for no, and 1 for yes")
    domainlist=[]
    rangelist=[]
    while NeedMap != 0 :
        domainselection = input("Enter number to select domain cell")
        rangeselection = input("Enter corresponding number to send {} to".format(nodedictionary1[domainselection]))
        domainlist.append(domainselection)
        rangelist.append(rangeselection)
        NeedMap=input("Any more functions to define? Select 1 for yes, select 2 for no")
    return domainlist, rangelist

#DrawSimplicialSet(MapValueAsGraph(2))

G=MapValueAsGraph(3)

def SimplicialGluing(SSetD, SSetR):
    n = max(len(SortByNodeAttribute(SSetD,'level')[1]),len(SortByNodeAttribute(SSetR,'level')[1]))
    domainlist=['[0]','[0, 0]','[0, 0, 0]']
    rangelist=['[1]','[1, 1]', '[1, 1, 1]']
    if n == len(SortByNodeAttribute(SSetD,'level')[1]):
        combinedlist=zip(domainlist, rangelist)
        mapping=dict(zip(domainlist,combinedlist))
        SSetD = nx.relabel_nodes(SSetD, mapping, copy=False)
        return SSetD
    if n == len(SortByNodeAttribute(SSetR,'level')[2]):
        combinedlist=zip(domainlist, rangelist)
        mapping=dict(zip(rangelist,combinedlist))
        SSetR = nx.relabel_nodes(SSetR, mapping, copy=False)
        return SSetD
    

DGLMaximal=dgl.from_networkx(G)

dgl.sampling.sample_neighbors(DGLMaximal)

print(DGLMaximal.adj_sparse(coo))