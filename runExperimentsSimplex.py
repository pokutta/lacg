import numpy as np
import datetime
import time
ts = time.time()
timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S').replace(' ', '-').replace(':', '-')

import os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6


from auxiliaryFunctions import project_onto_simplex, performUpdate, exitCriterion, stepSize


"""## Away FW or Pairwise FW"""
#Maintains active list of weights and vertices.
def runFWSimplex(x0, function, feasibleReg, tolerance, maxTime, FWVariant = "AFW", typeStep = "SS", criterion = "PG", criterionRef = 0.0):
    #Quantities we want to output.
    grad = function.fEvalGrad(x0)
    FWGap = [np.dot(grad, x0 - feasibleReg.LPOracle(grad))]
    fVal = [function.fEval(x0)]
    timing = [time.time()]
    x = x0.copy()
    itCount = 1
    while(True):
        if(FWVariant == "AFW"):
            x, vertvar, gap = awayStepFWSimplex(function, feasibleReg, x, typeStep)
        else:
            x, vertvar, gap = pairwiseStepFWSimplex(function, feasibleReg, x, typeStep)
        performUpdate(function, x, FWGap, fVal, timing, gap)
        if(exitCriterion(itCount, fVal[-1], FWGap[-1], criterion = criterion, numCriterion = tolerance, critRef = criterionRef) or timing[-1] - timing[0] > maxTime):
            timing[:] = [t - timing[0] for t in timing]
            return x, FWGap, fVal, timing
        itCount += 1
        
def awayStepFWSimplex(function, feasibleReg, x, typeStep):
    grad = function.fEvalGrad(x)
    v = feasibleReg.LPOracle(grad)
    a, indexMax = feasibleReg.AwayOracle(grad, x)    
    vertvar = 0
    #Choose FW direction, can overwrite index.
    if(np.dot(grad, x - v) > np.dot(grad, a - x)):
      d = v - x
      alphaMax = 1.0
      optStep = stepSize(function, d, grad, typeStep)
      alpha = min(optStep, alphaMax)
      #Less than maxStep
      if(alpha != alphaMax):
          #newVertex returns true if vertex is new.
          if(np.dot(v, x) == 0.0):
              vertvar = 1
      #Max step length away step, only one vertex now.
      else:
          vertvar = -1
    else:
      d = x - a
      alphaMax = x[indexMax]/(1.0 - x[indexMax])
      optStep = stepSize(function, d, grad, typeStep)
      alpha = min(optStep, alphaMax)
      #Max step, need to delete a vertex.
      if(alpha == alphaMax):
          vertvar = -1
    return x + alpha*d, vertvar, np.dot(grad, x - v)

#Perform one step of the Pairwise FW algorithm
#Also specifies if the number of vertices has decreased var = -1 or
#if it has increased var = +1. Otherwise 0.
def pairwiseStepFWSimplex(function, feasibleReg, x, typeStep):
    grad = function.fEvalGrad(x)
    v = feasibleReg.LPOracle(grad)
    a, index = feasibleReg.AwayOracle(grad, x)
    vertVar = 0
    #Find the weight of the extreme point a in the decomposition.
    alphaMax = x[index]
    #Update weight of away vertex.
    d = v - a
    optStep = stepSize(function, d, grad, typeStep)
    alpha = min(optStep, alphaMax)
    if(alpha == alphaMax):
        vertVar = -1
    #Update the FW vertex
    if(np.dot(v, x) == 0.0):
        vertVar = 1
    return x + alpha*d, vertVar, np.dot(grad, x - v)

        
"""
# LaCG Variants
"""
#Locally Accelerated Conditional Gradients. 
class LaCG:
    def run(self, x0, function, feasReg, tol, maxIter =  5e5, FWVariant = "AFW", typeStep = "SS"):
        #Perform lineseach?
        self.lineSearch = typeStep
        #Function parameters.
        self.restart = []
        self.L = function.largestEig()
        self.mu = function.smallestEig()
        self.tol = tol
        self.theta = np.sqrt(0.5*self.mu/self.L)
        #Copy the variables.
        self.xAFW, self.xAGD, x, self.y, self.w = [x0.copy(), x0.copy(), x0.copy(), x0.copy(), x0.copy()]
        #Store the data from the initial iterations.
        itCount = 1
        self.A = 1.0
        self.z = -function.fEvalGrad(self.xAFW) + self.L*self.xAFW
        #Initial data measurements.
        grad = function.fEvalGrad(x0)
        FWGap = [np.dot(grad, x0 - feasReg.LPOracle(grad))]
        fVal = [function.fEval(x0)]
        timing = [time.time()]
        while(fVal[-1] - fValOpt > tol):
            print(fVal[-1] - fValOpt)
            x, gap = self.runIter(function, feasReg, x, itCount + 1, FWVariant)
            performUpdate(function, x, FWGap, fVal, timing, gap)
            itCount += 1
            if(timing[-1] - timing[0] > TIME_LIMIT):
                break
        timing[:] = [t - timing[0] for t in timing]
        return x, FWGap, fVal, timing
    
    def runIter(self, function, feasReg, x, it, FWVariant):
        #Information about variation of active set in vertVar
        if(FWVariant == "AFW"):
            self.xAFW, vertVar, gap = awayStepFWSimplex(function, feasReg, x, typeStep = self.lineSearch)
        else:
            self.xAFW, vertVar, gap = pairwiseStepFWSimplex(function, feasReg, x, typeStep = self.lineSearch)
        self.xAGD = self.accelStep(function, x)
        #If we return the Accelerated point, the gap is invalid, set it to zero for later processing.
        if(function.fEval(self.xAGD) < function.fEval(self.xAFW)):
            return self.xAGD, gap
        else:
            return self.xAFW, gap

    #Whenever we perform an accelerated step, we can use a warm start for the
    #optimization subproblem, using w0 and alphaw0
    def accelStep(self, function, x):
        self.A = self.A/(1 - self.theta)
        a = self.theta*self.A
        self.y = (x + self.theta*self.w)/(1 + self.theta)
        self.z += a*(self.mu*self.y - function.fEvalGrad(self.y))
        #Compute the projection directly.
        indices = np.where(x > 0.0)[0]
        #Calculate the vector.
        b = self.z[indices]/(self.mu*self.A + self.L - self.mu)
        aux = project_onto_simplex(b)
        self.w = np.zeros(len(x))
        self.w[indices] = aux
        xAGD = (1 - self.theta)*x + self.theta*self.w
        return xAGD
        
#Takes an input scheme and tries to accelerate it.
#Need to specify function and scheme wich will be used for optimizing.
class catalystSchemeSimplex:
    def run(self, x0, function, feasReg, tol, maxTime, FWVariant = "AFW", typeStep = "SS"):
        self.L = function.largestEig()
        self.mu = function.smallestEig()
        self.kappa = self.L - 2*self.mu
        from collections import deque
        xOut = deque([x0], maxlen = 2)
        #Quantities we want to output.
        FWGap = [function.FWGapBaseProblem(xOut[-1], feasReg)]
        fVal = [function.fEvalBaseProblem(xOut[-1])]
        timing = [time.time()]
        iterations = [1]
        q = self.mu / (self.mu + self.kappa)
        rho = 0.9*np.sqrt(q)
        y = deque([x0, x0], maxlen = 2)
        function.setKappa(self.kappa)
        epsilon = 0.22222 * FWGap[-1] * (1-rho)
        alpha = deque([np.sqrt(q)], maxlen = 2)
        itCount = 0
        while(fVal[-1] - fValOpt > tol):
            function.sety(y[-1])
            newX, gap, fvalue, timingInner =  runFWSimplex(xOut[-1], function, feasReg, epsilon, maxTime/2.0, FWVariant = FWVariant, typeStep = typeStep, criterion = "DG")
            xOut.append(newX)
            epsilon *= (1-rho)
            iterations.append(len(gap) + iterations[-1])
            alpha.append(self.findRoot(alpha[-1], q))
            beta = self.returnBeta(alpha)
            y.append(xOut[-1] + beta *(xOut[-1] - xOut[-2]))
            performUpdate(function, xOut[-1], FWGap, fVal, timing, function.FWGapBaseProblem(xOut[-1], feasReg))
            if(timing[-1] - timing[0] > maxTime):
                break
            itCount += 1
        timing[:] = [t - timing[0] for t in timing]
        return xOut[-1], FWGap, fVal, timing, iterations
    
    #Finds the root of the equation between 0 and 1.
    #Throws an assertion if no valid candidate is found.
    def findRoot(self, alpha, q):
        aux = (q-alpha*alpha)
        val = 0.5*(aux + np.sqrt(aux*aux + 4.0*alpha*alpha))
        if(val > 0 and val <= 1):
            return val
        else:
            val = 0.5*(aux - np.sqrt(aux*aux + 4.0*alpha*alpha))
            assert val > 0 and val < 1, "Root does not meet desired criteria.\n"
            return val
        
    #Returns the value of Beta based on the values of alpha.
    #The alpha deque contains at least two values.
    def returnBeta(self, alpha):
        return alpha[-2]*(1-alpha[-2])/(alpha[-2]*alpha[-2] + alpha[-1])

"""# Simplex example"""
from functions import randomPSDGenerator, funcQuadratic, funcAccelScheme
from feasibleRegions import probabilitySimplexPolytope
from algorithms import NAGD_probabilitySimplex, CGS, DIPFW

TIME_LIMIT = int(1800)
size = int(1500)

feasibleRegion = probabilitySimplexPolytope(size)
x_0 = feasibleRegion.initialPoint()
S_0 = [x_0]
alpha_0 = [1]
tolerance = 1.0e-5
typeOfStep = "SS"

LVal = 1000.0
MuVal = 1.0
M = randomPSDGenerator(size, MuVal, LVal)
b = np.random.randint(-1,1, size = size)
fun = funcQuadratic(size, M , b, MuVal, LVal)

print("Solving the problem over the simplex.")
##Run to a high Frank-Wolfe primal gap accuracy for later use.
print("\nSolving the problem to a high accuracy using Nesterov's AGD to obtain a reference solution.")
fValOpt = NAGD_probabilitySimplex(x_0, fun, feasibleRegion, tolerance/10.0)

#Catalyst augmented
print("\nRunning Catalyst-augmented AFW.")
funCat = funcAccelScheme(len(x_0), fun.returnM(), fun.returnb(), fun.largestEig(), fun.smallestEig())
CatalystAFW = catalystSchemeSimplex()
xCatalyst, FWCatalyst, fCatalyst, tCatalyst, itCatalyst = CatalystAFW.run(x_0, funCat, feasibleRegion, tolerance, TIME_LIMIT)

##Vanilla AFW
print("\nRunning AFW.")
xAFW, FWGapAFW, fValAFW, timingAFW = runFWSimplex(x_0, fun, feasibleRegion, tolerance, TIME_LIMIT, FWVariant = "AFW", typeStep = typeOfStep, criterion = "PG", criterionRef = fValOpt)

##Vanilla PFW
print("\nRunning PFW.")
xPFW, FWGapPFW, fValPFW, timingPFW = runFWSimplex(x_0, fun, feasibleRegion, tolerance, TIME_LIMIT, FWVariant = "PFW", typeStep = typeOfStep, criterion = "PG", criterionRef = fValOpt)

#LaCG
print("\nRunning LaCG-AFW.")
LaCGAway = LaCG()
xLaCGAFW, FWGapLaCGAFW, fValLaCGAFW, timingLaCGAFW = LaCGAway.run(x_0, fun, feasibleRegion, tolerance, typeStep = typeOfStep, FWVariant = "AFW")

#LaCG PFW
print("\nRunning LaCG-PFW.")
LaCGPairwise = LaCG()
xLaCGPFW, FWGapLaCGPFW, fValLaCGPFW, timingLaCGPFW = LaCGPairwise.run(x_0, fun, feasibleRegion, tolerance, typeStep = typeOfStep, FWVariant = "PFW")

#Conditional Gradient Sliding.
print("\nRunning CGS.")
CGS = CGS()
xCGS, FWGapCGS, fValCGS, timingCGS, iterationCGS = CGS.run(x_0, fun, feasibleRegion, tolerance, TIME_LIMIT, criterion = "PG", criterionRef = fValOpt)

#Decomposition Invariant CG
print("\nRunning DICG.")
xDICG, FWGapDICG, fValDICG, timingDICG = DIPFW(x_0, fun, feasibleRegion, tolerance, TIME_LIMIT, typeStep = typeOfStep, criterion = "PG", criterionRef = fValOpt)

import matplotlib.pyplot as plt
#Plot primal gap in terms of iteration.
plt.loglog(np.arange(len(fValAFW)) + 1, [(x - fValOpt) for x in fValAFW], '-*', color = 'b', markevery = np.logspace(0, np.log10(len(fValAFW)-1), 10).astype(int).tolist(), label = 'AFW')
plt.loglog(np.arange(len(fValPFW)) + 1, [(x - fValOpt) for x in fValPFW], '-D', color = 'c',  markevery = np.logspace(0, np.log10(len(fValPFW)-1), 10).astype(int).tolist(), label = 'PFW')
plt.loglog(np.arange(len(fValLaCGAFW)) + 1, [(x - fValOpt) for x in fValLaCGAFW], '-o', color = 'k', markevery = np.logspace(0, np.log10(len(fValLaCGAFW)-1), 10).astype(int).tolist(), label = 'LaCG-AFW')
plt.loglog(np.arange(len(fValLaCGPFW)) + 1, [(x - fValOpt) for x in fValLaCGPFW], '-^', color = 'g', markevery = np.logspace(0, np.log10(len(fValLaCGPFW)-1), 10).astype(int).tolist(), label = 'LaCG-PFW')
plt.loglog(np.arange(len(fValDICG)) + 1, [(x - fValOpt) for x in fValDICG], '-s', color = 'r', markevery = np.logspace(0, np.log10(len(fValDICG)-1), 10).astype(int).tolist(), label = 'DICG')
plt.loglog(iterationCGS, [(x - fValOpt) for x in fValCGS],  color = 'y',  label = 'CGS')
plt.loglog(itCatalyst, [(x - fValOpt) for x in fCatalyst], ':', color = 'm',  label = 'Catalyst')
plt.legend()
plt.xlabel(r'$k$')
plt.ylabel(r'$f(x_{k}) - f^{*}$')
plt.autoscale(enable=True, axis='x', tight=True)
plt.grid()
plt.show()
plt.close()

#Plot Primal gap in terms of time.
plt.semilogy(timingAFW, [(x - fValOpt) for x in fValAFW], '-*', color = 'b', markevery = int(len(timingAFW)/10), label = 'AFW')
plt.semilogy(timingPFW, [(x - fValOpt) for x in fValPFW], '-D',  color = 'g', markevery = int(len(timingPFW)/10), label = 'PFW')
plt.semilogy(timingLaCGAFW, [(x - fValOpt) for x in fValLaCGAFW], '-o', color = 'k', markevery = int(len(timingLaCGAFW)/10), label = 'LaCG-AFW')
plt.semilogy(timingLaCGPFW, [(x - fValOpt) for x in fValLaCGPFW], '-^',  color = 'g', markevery = int(len(timingLaCGPFW)/10), label = 'LaCG-PFW')
plt.semilogy(timingDICG, [(x - fValOpt) for x in fValDICG], '-s', color = 'r', markevery = int(len(timingDICG)/10), label = 'DICG')
plt.semilogy(timingCGS, [(x - fValOpt) for x in fValCGS],   color = 'y', label = 'CGS')
plt.semilogy(tCatalyst, [(x - fValOpt) for x in fCatalyst], ':',  color = 'm', label = 'Catalyst')
plt.legend()
plt.ylabel(r'$f(x_{k}) - f^{*}$')
plt.xlabel(r't[s]')
plt.autoscale(enable=True, axis='x', tight=True)
plt.grid()
plt.show()
plt.close()