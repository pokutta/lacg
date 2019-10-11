import numpy as np
from auxiliaryFunctions import maxVertex

"""# LP Oracles"""

#Birkhoff Polytope feasible region.
class BirkhoffPolytope:
    def __init__(self, dim):
        self.dim = dim
        self.matdim = int(np.sqrt(dim))
        
    def LPOracle(self, x):
        from scipy.optimize import linear_sum_assignment
        objective = x.reshape((self.matdim, self.matdim))
        matching = linear_sum_assignment(objective)
        solution = np.zeros((self.matdim, self.matdim))
        solution[matching] = 1
        return solution.reshape(self.dim)
    
    def initialPoint(self):
        return np.identity(self.matdim).flatten()
    
    #Input is the vector over which we calculate the inner product.
    def AwayOracle(self, grad, activeVertex):
        return maxVertex(grad, activeVertex)

#Birkhoff Polytope feasible region.
class probabilitySimplexPolytope:
    def __init__(self, dim):
        self.dim = dim
        
    def LPOracle(self, x):
        v = np.zeros(len(x), dtype = float)
        v[np.argmin(x)] = 1.0
        return v
    
    #Input is the vector over which we calculate the inner product.
    def AwayOracle(self, grad, x):
        aux = np.multiply(grad, np.sign(x))
        indices = np.where(x > 0.0)[0]
        v = np.zeros(len(x), dtype = float)
        indexMax = indices[np.argmax(aux[indices])]
        v[indexMax] = 1.0
        return v, indexMax
    
    def initialPoint(self):
        v = np.zeros(self.dim)
        v[0] = 1.0
        return v
    
    
"""LP model based on Gurobi solver."""
from gurobipy import GRB, read, Column

###############################
# Algorithm Configuration
###############################
run_config = {
        'solution_only': True,
        'verbosity': 'normal',
        'OutputFlag': 0,
        'dual_gap_acc': 1e-06,
        'runningTimeLimit': None,
        'use_LPSep_oracle': True,
        'max_lsFW': 100000,
        'strict_dropSteps': True,
        'max_stepsSub': 100000,
        'max_lsSub': 100000,
        'LPsolver_timelimit': 100000,
        'K': 1
        }

"""## Gurobi LP Solver

Used for the MIPLIB and the Birkhoff Example.

"""
class GurobiPolytope:
    """LP model implemented via Gurobi."""
    def __init__(self, modelFilename, addCubeConstraints=False, transform_to_equality=False):
        model = read(modelFilename)
        model.setParam('OutputFlag', False)
        model.params.TimeLimit = run_config['LPsolver_timelimit']
        model.params.threads = 4
        model.params.MIPFocus = 0
        model.update()
        if addCubeConstraints:
            counter = 0
            for v in model.getVars():
                model.addConstr(v <= 1, 'unitCubeConst' + str(counter))
                counter += 1
        model.update()
        if transform_to_equality:
            for c in model.getConstrs():
                sense = c.sense
                if sense == GRB.GREATER_EQUAL:
                    model.addVar(obj=0, name="ArtN_" + c.constrName, column=Column([-1], [c]))
                if sense == GRB.LESS_EQUAL:
                    model.addVar(obj=0, name="ArtP_" + c.constrName, column=Column([1], [c]))
                c.sense = GRB.EQUAL
        model.update()
        self.dimension = len(model.getVars())
        self.model = model
        return 

    """
    To find the total number of constraints in a model: model.NumConstrs
    To return the constraints of a model: model.getConstrs()
    To add a single constraint to the model model.addConstr(model.getVars()[-1] == 0, name = 'newConstraint1')
    If we want to delete the last constraint that was added we do: model.remove(model.getConstrs()[-1])
    """
    def LPOracle(self, cc):
        """Find good solution for cc with optimality callback."""
        m = self.model
        for it, v in enumerate(m.getVars()):
            v.setAttr(GRB.attr.Obj, cc[it])
        #Update the model with the new atributes.
        m.update()
        m.optimize(lambda mod, where: fakeCallback(mod, where, GRB.INFINITY))
        # Status checking
        status = m.getAttr(GRB.Attr.Status)
        if status == GRB.INF_OR_UNBD or \
           status == GRB.INFEASIBLE  or \
           status == GRB.UNBOUNDED:
            assert False, "The model cannot be solved because it is infeasible or unbounded"
        if status != GRB.OPTIMAL:
            print(status)
            assert False, "Optimization was stopped."
        #Store the solution that will be outputted.
        solution = np.array([v.x for v in m.getVars()], dtype=float)[:]
        #Check that the initial number of constraints and the final number is the same.
        return solution
    
    def initialPoint(self):
        print("Finding Initial Point.")
        return self.LPOracle(np.zeros(self.dimension))
     
    #Input is the vector over which we calculate the inner product.
    def AwayOracle(self, grad, activeVertex):
        return maxVertex(grad, activeVertex)
        
    def dim(self):
         return self.dimension
        
def fakeCallback(model, where, value):
    ggEps = 1e-08
    if where == GRB.Callback.MIPSOL:
        # x = model.cbGetSolution(model.getVars())
        # logging.info 'type of x: ' + str(type(x))
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if obj < value - ggEps:
            print('early termination with objective value :{}'.format(obj))
            print('which is better than {}'.format(value - ggEps))
            # model.terminate()

    if where == GRB.Callback.MIP:
        objBnd = model.cbGet(GRB.Callback.MIP_OBJBND)

        if objBnd >= value + ggEps:
            # model.terminate()
            pass


import networkx as nx
import matplotlib.pyplot as plt
import math

#Randomly generated Graphs
#p is the redirection probability.
def generateRandomGraph(n, p):
    DG = nx.gnr_graph(n, p)
    return DG
    
#Two vertices in the end.
#m represents the number of layers
#s represents the number of nodes per layer.
def generateStructuredGraph(layers, nodesPerLayer):
    m = layers
    s = nodesPerLayer
    DG = nx.DiGraph()
    DG.add_nodes_from(range(0, m*s+ 1))
    #Add first edges between source
    DG.add_edges_from([(0,x + 1) for x in range(s)])  
    #Add all the edges in the subsequent layers.
    for i in range(m - 1):
        DG.add_edges_from([(x + 1 + s*i , y + 1 + s*(i + 1)) for x in range(s) for y in range(s)])
    DG.add_edges_from([( x + 1 + s*(m - 1) , m*s + 1) for x in range(s)])
    return DG

"""
If typegraph = "Structured":
    param1 = number of layers
    param2 = number of nodes per layer.
    
Otherwise:
Growing network with redirection (GNR) digraph
    param1 = number of nodes
    param2 = The redirection probability.
    
Can draw the graph with the command nx.draw()
"""
class flowPolytope:
    """Shortest path problem on a DAG."""
    def __init__(self, param1, param2, typeGraph = "Structured"):
        #Generate the type of graph that we want
        if(typeGraph == "Structured"):
            self.graph = generateStructuredGraph(param1, param2)
        else:
            self.graph = generateRandomGraph(param1, param2)
        #Sort the graph in topological order
        self.topologicalSort = list(nx.topological_sort(self.graph))
        self.dictIndices = self.constructDictionaryIndices(self.graph)
        return 
        
    #Given a graph,a dictionary, a set of weights and a topological order.
    def LPOracle(self,  weight):
        d = math.inf*np.ones(nx.number_of_nodes(self.graph)) 
        d[self.topologicalSort[0]] = 0.0
        p = -np.ones(nx.number_of_nodes(self.graph), dtype = int)
        for u in self.topologicalSort:
            for v in self.graph.neighbors(u):
                self.relax(u, v, d, weight, p)
                
        pathAlg = [self.topologicalSort[-1]]
        while(pathAlg[-1] != self.topologicalSort[0]):
            pathAlg.append(p[pathAlg[-1]])
        pathAlg.reverse()
        #Reconstruc the vertex.
        outputVect = np.zeros(nx.number_of_edges(self.graph))
        for i in range(len(pathAlg) - 1):
            outputVect[self.dictIndices[(pathAlg[i], pathAlg[i + 1])]] = 1.0
        return outputVect
    
    def relax(self, i, j, dVect, wVect, pVect):
        if dVect[j] > dVect[i] + wVect[self.dictIndices[(i, j)]]:
            dVect[j] = dVect[i] + wVect[self.dictIndices[(i, j)]]
            pVect[j] = i
        return
        
    #Bellman-Ford algorithm for shortest path.
    def LPOracleBellmanFord(self,  weight):
        self.weight = weight.copy()
        pathAlg = nx.bellman_ford_path(self.graph, self.topologicalSort[0], self.topologicalSort[-1], self.func)
        #Reconstruct the vertex.
        outputVect = np.zeros(nx.number_of_edges(self.graph))
        for i in range(len(pathAlg) - 1):
            outputVect[self.dictIndices[(pathAlg[i], pathAlg[i + 1])]] = 1.0
        return outputVect
 
    #Function that returns the values of the weights.
    def func(self, u, v, wVect):
        return self.weight[self.dictIndices[(v, u)]]
    
    #Given a DAG, returns a mapping from the edges to indices from 0 to N
    #where N represents the number of Edges.
    def constructDictionaryIndices(self, graph):
        #Construct a dictionary of the indices
        dictionary = {}
        itCount = 0
        for i in graph.edges:
            dictionary[i] = itCount
            itCount += 1
        return dictionary

        def dim(self):
         return self.dimension
     
    def initialPoint(self):
        print("Finding Initial Point.")
        aux = np.random.rand(self.dim())
        return self.LPOracle(aux)
     
    def dim(self):
         return self.graph.number_of_edges()
     
    def plot(self):
         nx.draw(self.graph)
         plt.show()
         
    def returnEdges(self):
         return self.graph.edges()
     
    def topologicalOrdering(self):
         return self.topologicalSort
     
    #Input is the vector over which we calculate the inner product.
    def AwayOracle(self, grad, activeVertex):
        return maxVertex(grad, activeVertex)