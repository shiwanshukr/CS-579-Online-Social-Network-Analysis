import networkx as nx
import numpy as np





def plot_graph():

    """
    Plot this subgraph of nodes, coloring
    the specified list of target_nodes in red.
    """
    graph = nx.Graph()
    graph.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'H'), ('B', 'D'), ('C', 'G'), ('C', 'D'), ('D', 'E'), ('D', 'F')])

    nx.draw(graph)
    #plt.figure(figsize=(10,10))
    #plt.axis('off')
    plt.show()


def jaccard_wt(graph, node):
    """
    The weighted jaccard score, defined above.
    Args:
      graph....a networkx graph
      node.....a node to score potential new edges for.
    Returns:
      A list of ((node, ni), score) tuples, representing the 
                score assigned to edge (node, ni)
                (note the edge order)
    """
    # Create a list where nodes and jaccard score is stored. jaccard_score[] : 

    jaccard_score = []

    # All the nodes which are not predefined 

    entrant_node = set(graph.nodes()) - set(graph.neighbors(node)) - {node}

    graph_degree = graph.degree()

    neighbours_node = set(graph.neighbors(node))

    for ni in entrant_node:

      jaccard_score.append(((node, ni), jaccard_value(graph_degree, neighbours_node, set(graph.neighbors(ni)))))

    return sorted(jaccard_score, key=lambda x: (-x[1], x[0][1]))



def jaccard_value(graph_degree, first_neighbour_set_A, second_neighbour_set_B):

    # Where A and B are set of neighbours  of two nodes to be scored.
    summation_intersection_A_B = []

    common_nodes = first_neighbour_set_A.intersection(second_neighbour_set_B)

    for n in common_nodes:

      summation_intersection_A_B.append(1 / graph_degree[n])

    #summation_intersection_A_B = [1 / graph_degree[n] for n in common_nodes]


    # first_neighbour_set_A
    summation_A = [graph_degree[nodei] for nodei in first_neighbour_set_A]

    # second_neighbour_set_B 
    summation_B = [graph_degree[nodei] for nodei in second_neighbour_set_B]

    '''
    calculations :

    value = summation_intersection_A_B/ first_neighbour_set_A + second_neighbour_set_B

    '''
    summation_numerator =  np.sum(summation_intersection_A_B)

    summation_denominator = ((1 / np.sum( summation_A ) ) + (1 / np.sum( summation_B )))

    result = summation_numerator / summation_denominator

    return result

'''

#  ✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦ TO CHECK ✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦
# I have created this main to check the efficiency of the program and the output of the file is : 
  ✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦ Pring Jaccard Score ✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦

[(('A', 'D'), 2.285714285714286), (('A', 'G'), 0.6666666666666666), (('A', 'H'), 0.6666666666666666), (('A', 'E'), 0.0), (('A', 'F'), 0.0)]
  ✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦ Printing Graph ✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦
  for given set of data :
  graph = nx.Graph()
  graph.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'H'), ('B', 'D'), ('C', 'G'), ('C', 'D'), ('D', 'E'), ('D', 'F')])
'''



# ✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦ MAIN ✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦

'''
if __name__ == '__main__':

  graph = trial_Graph()

  print("\t✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦ Pring Jaccard Score ✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦\n")


  print(jaccard_wt(graph, 'A'))


  print("\t✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦ Printing Graph ✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦\n")

  #plot_graph()

  #deg = graph.degree()
  #mentor = sorted(deg,key=lambda x: x[-1],reverse=True)[1]
  #subgraph = get_subgraph(graph, [mentor])
  #plot_subgraph(graph, [mentor])

'''

