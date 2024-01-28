import dgl
import collections
from KanofDelta import KanofDelta

class AdjFunctor():
    def __init__(self,simplicial_set):
        self.simplicial_set = simplicial_set
        self.adj_graph_node_dict = dict()
        self.adj_graph_simplex_ids = dict()
        self.node_index = 0
        for list_of_simplices in self.simplicial_set.values():
            for simplex in list_of_simplices:
                self.adj_graph_node_dict.update({self.node_index: simplex})
                self.adj_graph_simplex_ids.update({id(simplex):self.node_index})
                self.node_index = self.node_index + 1
        self.dimensions      = list(self.simplicial_set.keys())
        self.max_dim         = max(self.dimensions)
        self.adj_graph = dgl.heterograph(
            {('degenerate', 's', 'degenerate'): ([], []),
             ('degenerate', 'd', 'non-degenerate'): ([], []),
             ('non-degenerate', 'd', 'non-degenerate'): ([], []),
             ('non-degenerate', 's', 'degenerate'): ([], []),
             ('degenerate', 'd', 'degenerate'): ([], [])
             })
               
    def MappingPossibility(self,possiblesublist, possiblesuperlist):
        """
        checks if one list is a sublist of the other by simply
        counting the number of entries
        """
        c1 = collections.Counter(possiblesublist)
        c2 = collections.Counter(possiblesuperlist)
        for item, counted in c1.items():
            if counted > c2[item]:
               return False
        return True

    def fill_edges(self):
        for d in range(self.max_dim-1):
            for u_simplex in self.simplicial_set[d+1]:
                for l_simplex in self.simplicial_set[d]:
                    if self.MappingPossibility(l_simplex, u_simplex):
                        u_node = self.adj_graph_simplex_ids[id(u_simplex)]
                        l_node = self.adj_graph_simplex_ids[id(l_simplex)]
                        u_deg  = False
                        l_deg  = False
                        u_ndeg = False
                        l_ndeg = False
                        #check if u_simplex is nondegenerate
                        if len(u_simplex) != len(set(u_simplex)):
                            #self.adj_graph.add_nodes(u_node, ntype='degenerate')
                            u_deg = True
                        else:
                            #self.adj_graph.add_nodes(u_node, ntype='non-degenerate')
                            u_ndeg = True
                        #check if l_simplex i nondegenerate
                        if len(l_simplex) == len(set(l_simplex)):
                            l_ndeg = True
                            #self.adj_graph.add_nodes(l_node, ntype='non-degenerate')
                        else:
                            #self.adj_graph.add_nodes(l_node, ntype='degenerate')
                            l_deg = True
                        if u_deg:
                            if l_deg:
                                self.adj_graph.add_edges(l_node,u_node, etype=('degenerate','s','degenerate'))
                                self.adj_graph.add_edges(u_node,l_node, etype=('degenerate','d','degenerate'))
                            else:
                                self.adj_graph.add_edges(l_node,u_node, etype=('non-degenerate','s','degenerate'))
                                self.adj_graph.add_edges(u_node,l_node, etype=('degenerate','d','non-degenerate'))
                        if u_ndeg:
                            if l_ndeg:
                                self.adj_graph.add_edges(u_node,l_node, etype=('non-degenerate','d','non-degenerate'))
            
delta = {0:[[0],[1],[2]],1:[[0,1],[0,2],[1,2]],2:[[0,1,2]]}
expansion=KanofDelta(delta)
expansion.universal_completion()
graph = AdjFunctor(expansion.simplicial_set)
graph.fill_edges()
print(graph.adj_graph_node_dict)
print(graph.adj_graph.edges(etype=('non-degenerate','s','degenerate')))