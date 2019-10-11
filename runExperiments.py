if __name__== "__main__":
    
    import os
    #Computing parameters.
    os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
    os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
    
    #General imports
    import numpy as np
    import os
    import time
    import datetime
    import matplotlib.pyplot as plt
    from scipy.sparse import csc_matrix
    from algorithms import CGS, LaCG, runFW, DIPFW, catalystScheme
    from functions import funcQuadraticDiag, funcAccelSchemeDiag, funcAccelScheme, funcQuadratic
    from auxiliaryFunctions import randomPSDGeneratorSparse, calculateEigenvalues

    
    """
    ------------------------------BIRKHOFF POLYTOPE----------------------------
    """
    
    from feasibleRegions import BirkhoffPolytope
    
    #Time limit spent calculating the reference solution and running the algorithm.
    TIME_LIMIT_REFERENCE_SOL = int(4*3600)
    TIME_LIMIT = int(1800)
    size = int(40*40)
    
    feasibleRegion = BirkhoffPolytope(size)
    x_0 = feasibleRegion.initialPoint()
    S_0 = [x_0]
    alpha_0 = [1]
    tolerance = 1.0e-2
    typeOfStep = "SS"
    
    #Define the objective function.    
    sparsity = 0.01
    M  = randomPSDGeneratorSparse(size, sparsity)
    L, Mu = calculateEigenvalues(M)
    Matrix = csc_matrix(M)
    LVal = L
    MuVal = Mu
    b = np.zeros(size)
    fun = funcQuadratic(size, Matrix, b, Mu, L)
    
    print("Solving the problem over the Birkhoff polytope.")
    
    ##Run to a high Frank-Wolfe primal gap accuracy for later use.
    print("\nFinding optimal solution to high accuracy using DIPFW or Lazy AFW.")
    xTest, FWGapTest, fValTest, timingTest, activeSetTest = runFW(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance/2.0, TIME_LIMIT_REFERENCE_SOL, FWVariant = "Lazy", typeStep = "EL", criterion = "DG")
    fValOpt = fValTest[-1]
    tolerance = min(np.asarray(FWGapTest))
    
    #LaCG
    print("\nRunning LaCG-AFW.")
    LaCGAway = LaCG()
    xLaCGAFW, FWGapLaCGAFW, fValLaCGAFW, timingLaCGAFW, activeLaCGAFW = LaCGAway.run(x_0, fun, feasibleRegion, tolerance, TIME_LIMIT, FWVariant = "AFW", typeStep = typeOfStep, criterionRef = fValOpt)
    restartsLaCG = LaCGAway.returnRestarts()
    restartsLaCG[:] = np.asarray([x - 1 for x in restartsLaCG])
    
    #LaCG Lazy
    print("\nRunning LaCG-AFW Lazy.")
    LaCGAwayLazy = LaCG()
    xLaCGAFWLazy, FWGapLaCGAFWLazy, fValLaCGAFWLazy, timingLaCGAFWLazy, activeLaCGAFWLazy = LaCGAwayLazy.run(x_0, fun, feasibleRegion, tolerance, TIME_LIMIT, FWVariant = "Lazy", typeStep = typeOfStep, criterionRef = fValOpt)
    restartsLaCGLazy = LaCGAwayLazy.returnRestarts()
    restartsLaCGLazy[:] = np.asarray([x - 1 for x in restartsLaCGLazy])
    
    #LaCG PFW
    print("\nRunning LaCG-PFW.")
    LaCGPairwise = LaCG()
    xLaCGPFW, FWGapLaCGPFW, fValLaCGPFW, timingLaCGPFW, activeLaCGPFW = LaCGPairwise.run(x_0, fun, feasibleRegion, tolerance, TIME_LIMIT, FWVariant = "PFW", typeStep = typeOfStep, criterionRef = fValOpt)
    restartsLaCGPFW = LaCGPairwise.returnRestarts()
    restartsLaCGPFW[:] = np.asarray([x - 1 for x in restartsLaCGPFW])
    
    #Vanilla PFW
    print("\nRunning PFW.")
    xPFW, FWGapPFW, fValPFW, timingPFW, activeSetAFW = runFW(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance, TIME_LIMIT, FWVariant = "PFW", typeStep = typeOfStep, criterion = "PG", criterionRef = fValOpt)
    
    #Vanilla AFW
    print("\nRunning AFW.")
    xAFW, FWGapAFW, fValAFW, timingAFW, activeSetPFW = runFW(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance, TIME_LIMIT, FWVariant = "AFW", typeStep = typeOfStep, criterion = "PG", criterionRef = fValOpt)
    
    #Run Lazy AFW
    print("\nRunning Lazy AFW.")
    xAFWLazy, FWGapAFWLazy, fValAFWLazy, timingAFWLazy, activeSetLazy  = runFW(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance, TIME_LIMIT, FWVariant = "Lazy", typeStep = typeOfStep, criterion = "PG", criterionRef = fValOpt)
    
    #Decomposition Invariant CG
    print("\nRunning DICG.")
    xDICG, FWGapDICG, fValDICG, timingDICG = DIPFW(x_0, fun, feasibleRegion, tolerance, TIME_LIMIT, typeStep = typeOfStep, criterion = "PG", criterionRef = fValOpt)

    #Catalyst augmented
    print("\nRunning Catalyst-augmented AFW.")
    funCat = funcAccelScheme(len(x_0), fun.returnM(), fun.returnb(), fun.largestEig(), fun.smallestEig())
    CatalystAFW = catalystScheme()
    xCatalyst, FWCatalyst, fCatalyst, tCatalyst, itCatalyst = CatalystAFW.run(x_0, funCat, feasibleRegion, tolerance, TIME_LIMIT, FWVariant = "AFW", typeStep = "SS", criterionRef = fValOpt)
    
    #Conditional Gradient Sliding.
    print("\nRunning CGS.")
    CGS = CGS()
    xCGS, FWGapCGS, fValCGS, timingCGS, iterationCGS = CGS.run(x_0, fun, feasibleRegion, tolerance, TIME_LIMIT, criterion = "PG", criterionRef = fValOpt)
    
    #Plot primal gap in terms of iteration.
    plt.loglog(np.arange(len(fValAFW)), [(x - fValOpt) for x in fValAFW], '-*', color = 'b', markevery = np.logspace(0, np.log10(len(fValAFW)-1), 10).astype(int).tolist(), label = 'AFW')
    plt.loglog(np.arange(len(fValPFW)), [(x - fValOpt) for x in fValPFW], '-D', color = 'c',  markevery = np.logspace(0, np.log10(len(fValPFW)-1), 10).astype(int).tolist(), label = 'PFW')
    plt.loglog(np.arange(len(fValAFWLazy)), [(x - fValOpt) for x in fValAFWLazy], '--*', color = 'b', markerfacecolor='none', markevery = np.logspace(0, np.log10(len(fValAFWLazy)-1), 10).astype(int).tolist(), label = 'AFW (L)')
    plt.loglog(np.arange(len(fValLaCGAFW)), [(x - fValOpt) for x in fValLaCGAFW], '-o', color = 'k', markevery = np.logspace(0, np.log10(len(fValLaCGAFW)-1), 10).astype(int).tolist(), label = 'LaCG-AFW')
    plt.loglog(np.arange(len(fValLaCGPFW)), [(x - fValOpt) for x in fValLaCGPFW], '-^', color = 'g', markevery = np.logspace(0, np.log10(len(fValLaCGPFW)-1), 10).astype(int).tolist(), label = 'LaCG-PFW')
    plt.loglog(np.arange(len(fValLaCGAFWLazy)), [(x - fValOpt) for x in fValLaCGAFWLazy], '--o', color = 'k', markerfacecolor='none', markevery = np.logspace(0, np.log10(len(fValLaCGAFWLazy)-1), 10).astype(int).tolist(), label = 'LaCG-AFW (L)')
    plt.loglog(np.arange(len(fValDICG)), [(x - fValOpt) for x in fValDICG], '-s', color = 'r', markevery = np.logspace(0, np.log10(len(fValDICG)-1), 10).astype(int).tolist(), label = 'DICG')
    plt.loglog(iterationCGS, [(x - fValOpt) for x in fValCGS],  color = 'y',  label = 'CGS')
    plt.loglog(itCatalyst, [(x - fValOpt) for x in fCatalyst], ':', color = 'm',  label = 'Catalyst')
    plt.legend()
    plt.xlabel(r'$k$')
    plt.ylabel(r'$f(x_{k}) - f^{*}$')
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
    plt.grid()
    plt.show()
    plt.close()
    
#    #Generate a timestamp for the example and save data
#    ts = time.time()
#    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S').replace(' ', '-').replace(':', '-')
#    with open(os.path.join(os.getcwd(), "Birkhoff_" + str(timestamp) + "_Mu" + str(MuVal) + "_L" + str(LVal)+ "_Size" + str(size)  + "_TypeStep_" + typeOfStep  + ".txt"), 'w') as f:
#        f.write("size:\t" + str(len(x_0)) + "\n")
#        f.write("Mu:\t" + str(fun.smallestEig())+ "\n")
#        f.write("LVal:\t" + str(fun.largestEig())+ "\n")
#        f.write("Tolerance:\t" + str(tolerance)+ "\n")
#        f.write("Optimum:\t" + str(fValOpt)+ "\n")
#        #Output the FW Gap.
#        f.write("AFW"+ "\n")
#        f.write(str([x - fValOpt for x in fValAFW]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValAFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapAFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingAFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(activeSetAFW).replace("[", "").replace("]", "") + "\n")
#        #Output the FW Gap.
#        f.write("PFW"+ "\n")
#        f.write(str([x - fValOpt for x in fValPFW]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValPFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapPFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingPFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(activeSetPFW).replace("[", "").replace("]", "") + "\n")
#        #Output the FW Gap.
#        f.write("Lazy AFW"+ "\n")
#        f.write(str([x - fValOpt for x in fValAFWLazy]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write(str(activeSetLazy).replace("[", "").replace("]", "") + "\n")
#        f.write("LaCG-AFW"+ "\n")
#        f.write(str([x - fValOpt for x in fValLaCGAFW]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValLaCGAFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapLaCGAFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingLaCGAFW) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(restartsLaCG) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(activeLaCGAFW).replace("[", "").replace("]", "") + "\n")
#        f.write("LaCG-PFW"+ "\n")
#        f.write(str([x - fValOpt for x in fValLaCGPFW]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValLaCGPFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapLaCGPFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingLaCGPFW) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(restartsLaCGPFW) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(activeLaCGPFW).replace("[", "").replace("]", "") + "\n")
#        f.write("LaCG-AFW Lazy"+ "\n")
#        f.write(str([x - fValOpt for x in fValLaCGAFWLazy]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValLaCGAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapLaCGAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingLaCGAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write(str(restartsLaCGLazy) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(activeLaCGAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write("CGS"+ "\n")
#        f.write(str([x - fValOpt for x in fValCGS]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValCGS).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapCGS).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingCGS) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(iterationCGS) .replace("[", "").replace("]", "") + "\n")
#        f.write("DICG"+ "\n")
#        f.write(str([x - fValOpt for x in fValDICG]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValDICG).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapDICG).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingDICG) .replace("[", "").replace("]", "") + "\n")
#        f.write("Catalyst"+ "\n")
#        f.write(str([x - fValOpt for x in fCatalyst]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fCatalyst).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWCatalyst).replace("[", "").replace("]", "") + "\n")
#        f.write(str(tCatalyst) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(itCatalyst) .replace("[", "").replace("]", "") + "\n")
    
    
    """
    ------------------------------MIPLIB POLYTOPE----------------------------
    """
    from feasibleRegions import GurobiPolytope
    from auxiliaryFunctions import randomPSDGenerator
    
    #Good Instance
    file = 'ran14x18-disj-8.mps'
    TIME_LIMIT_REFERENCE_SOL = int(1800)
    TIME_LIMIT = int(800)
    MuVal = 0.1
    LVal = 10.0
    typeOfStep = "SS"
    tolerance = 0.1
    size = 504
    
    #Create the objective function.
    file = 'ran14x18-disj-8.mps'
    matrix = randomPSDGenerator(size, MuVal, LVal)
    b = np.zeros(size)
    fun = funcQuadratic(size, matrix, b, MuVal, LVal)

    pathfile = os.path.join(os.getcwd(), 'MIPLIB', file)
    feasibleRegion = GurobiPolytope(pathfile)
    size = feasibleRegion.dim()
    
    #Initial point
    x_0 = feasibleRegion.initialPoint()
    S_0 = [x_0]
    alpha_0 = [1]
    
    ##Run to a high Frank-Wolfe primal gap accuracy for later use.
    print("Finding optimal solution to high accuracy using DIPFW or Lazy AFW.")
    xTest, FWGapTest, fValTest, timingTest, activeSetTest = runFW(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance/2.0, TIME_LIMIT_REFERENCE_SOL, FWVariant = "Lazy", typeStep = "EL", criterion = "DG")
    fValOpt = fValTest[-1]
    tolerance = min(np.asarray(FWGapTest))
    
    #Conditional Gradient Sliding.
    print("Running CGS.")
    CGS = CGS()
    xCGS, FWGapCGS, fValCGS, timingCGS, iterationCGS = CGS.run(x_0, fun, feasibleRegion, tolerance, TIME_LIMIT, criterion = "PG", criterionRef = fValOpt)
    
    #Catalyst augmented
    print("Running Catalyst-augmented AFW.")
    funCat = funcAccelScheme(len(x_0), fun.returnM(), fun.returnb(), fun.largestEig(), fun.smallestEig())
    CatalystAFW = catalystScheme()
    xCatalyst, FWCatalyst, fCatalyst, tCatalyst, itCatalyst = CatalystAFW.run(x_0, funCat, feasibleRegion, tolerance, TIME_LIMIT)
    
    #LaCG
    print("Running LaCG-AFW.")
    LaCGAway = LaCG()
    xLaCGAFW, FWGapLaCGAFW, fValLaCGAFW, timingLaCGAFW, activeLaCGAFW = LaCGAway.run(x_0, fun, feasibleRegion, tolerance, TIME_LIMIT, FWVariant = "AFW", typeStep = typeOfStep, criterionRef = fValOpt)
    restartsLaCG = LaCGAway.returnRestarts()
    restartsLaCG[:] = np.asarray([x - 1 for x in restartsLaCG])
    
    #LaCG Lazy
    print("Running LaCG-AFW Lazy.")
    LaCGAwayLazy = LaCG()
    xLaCGAFWLazy, FWGapLaCGAFWLazy, fValLaCGAFWLazy, timingLaCGAFWLazy, activeLaCGAFWLazy = LaCGAwayLazy.run(x_0, fun, feasibleRegion, tolerance, TIME_LIMIT, FWVariant = "Lazy", typeStep = typeOfStep, criterionRef = fValOpt)
    restartsLaCGLazy = LaCGAwayLazy.returnRestarts()
    restartsLaCGLazy[:] = np.asarray([x - 1 for x in restartsLaCGLazy])
    
    #LaCG PFW
    print("Running LaCG-PFW.")
    LaCGPairwise = LaCG()
    xLaCGPFW, FWGapLaCGPFW, fValLaCGPFW, timingLaCGPFW, activeLaCGPFW = LaCGPairwise.run(x_0, fun, feasibleRegion, tolerance, TIME_LIMIT, FWVariant = "PFW", typeStep = typeOfStep, criterionRef = fValOpt)
    restartsLaCGPFW = LaCGPairwise.returnRestarts()
    restartsLaCGPFW[:] = np.asarray([x - 1 for x in restartsLaCGPFW])
    
    #Vanilla PFW
    print("Running PFW.")
    xPFW, FWGapPFW, fValPFW, timingPFW, activeSetAFW = runFW(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance, TIME_LIMIT, FWVariant = "PFW", typeStep = typeOfStep, criterion = "PG", criterionRef = fValOpt)
    
    #Vanilla AFW
    print("Running AFW.")
    xAFW, FWGapAFW, fValAFW, timingAFW, activeSetPFW = runFW(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance, TIME_LIMIT, FWVariant = "AFW", typeStep = typeOfStep, criterion = "PG", criterionRef = fValOpt)
    
    #Run Lazy AFW
    print("Running Lazy AFW.")
    xAFWLazy, FWGapAFWLazy, fValAFWLazy, timingAFWLazy, activeSetLazy  = runFW(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance, TIME_LIMIT, FWVariant = "Lazy", typeStep = typeOfStep, criterion = "PG", criterionRef = fValOpt)
    
    #Plot primal gap in terms of iteration.
    plt.loglog(np.arange(len(fValAFW)), [(x - fValOpt) for x in fValAFW], '-*', color = 'b', markevery = np.logspace(0, np.log10(len(fValAFW)-1), 10).astype(int).tolist(), label = 'AFW')
    plt.loglog(np.arange(len(fValPFW)), [(x - fValOpt) for x in fValPFW], '-D', color = 'c',  markevery = np.logspace(0, np.log10(len(fValPFW)-1), 10).astype(int).tolist(), label = 'PFW')
    plt.loglog(np.arange(len(fValAFWLazy)), [(x - fValOpt) for x in fValAFWLazy], '--*', color = 'b', markerfacecolor='none', markevery = np.logspace(0, np.log10(len(fValAFWLazy)-1), 10).astype(int).tolist(), label = 'AFW (L)')
    plt.loglog(np.arange(len(fValLaCGAFW)), [(x - fValOpt) for x in fValLaCGAFW], '-o', color = 'k', markevery = np.logspace(0, np.log10(len(fValLaCGAFW)-1), 10).astype(int).tolist(), label = 'LaCG-AFW')
    plt.loglog(np.arange(len(fValLaCGPFW)), [(x - fValOpt) for x in fValLaCGPFW], '-^', color = 'g', markevery = np.logspace(0, np.log10(len(fValLaCGPFW)-1), 10).astype(int).tolist(), label = 'LaCG-PFW')
    plt.loglog(np.arange(len(fValLaCGAFWLazy)), [(x - fValOpt) for x in fValLaCGAFWLazy], '--o', color = 'k', markerfacecolor='none', markevery = np.logspace(0, np.log10(len(fValLaCGAFWLazy)-1), 10).astype(int).tolist(), label = 'LaCG-AFW (L)')
    plt.loglog(iterationCGS, [(x - fValOpt) for x in fValCGS],  color = 'y',  label = 'CGS')
    plt.loglog(itCatalyst, [(x - fValOpt) for x in fCatalyst], ':', color = 'm',  label = 'Catalyst')
    plt.legend()
    plt.xlabel(r'$k$')
    plt.ylabel(r'$f(x_{k}) - f^{*}$')
    plt.grid()
    plt.show()
    plt.close()
    
    #Plot Primal gap in terms of time.
    plt.semilogy(timingAFW, [(x - fValOpt) for x in fValAFW], '-*', color = 'b', markevery = int(len(timingAFW)/10), label = 'AFW')
    plt.semilogy(timingPFW, [(x - fValOpt) for x in fValPFW], '-D',  color = 'g', markevery = int(len(timingPFW)/10), label = 'PFW')
    plt.semilogy(timingLaCGAFW, [(x - fValOpt) for x in fValLaCGAFW], '-o', color = 'k', markevery = int(len(timingLaCGAFW)/10), label = 'LaCG-AFW')
    plt.semilogy(timingLaCGPFW, [(x - fValOpt) for x in fValLaCGPFW], '-^',  color = 'g', markevery = int(len(timingLaCGPFW)/10), label = 'LaCG-PFW')
    plt.semilogy(timingCGS, [(x - fValOpt) for x in fValCGS],   color = 'y', label = 'CGS')
    plt.semilogy(tCatalyst, [(x - fValOpt) for x in fCatalyst], ':',  color = 'm', label = 'Catalyst')
    plt.legend()
    plt.ylabel(r'$f(x_{k}) - f^{*}$')
    plt.xlabel(r't[s]')
    plt.grid()
    plt.show()
    plt.close()
    
#    #Generate a timestamp for the example and save data
#    ts = time.time()
#    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S').replace(' ', '-').replace(':', '-')
#    with open(os.path.join(os.getcwd(), "MIPLIB_" + str(timestamp) + "_Mu" + str(MuVal) + "_L" + str(LVal)+ "_Size" + str(size)  + "_TypeStep_" + typeOfStep  + ".txt"), 'w') as f:
#        f.write("size:\t" + str(len(x_0)) + "\n")
#        f.write("Mu:\t" + str(fun.smallestEig())+ "\n")
#        f.write("LVal:\t" + str(fun.largestEig())+ "\n")
#        f.write("Tolerance:\t" + str(tolerance)+ "\n")
#        f.write("Optimum:\t" + str(fValOpt)+ "\n")
#        #Output the FW Gap.
#        f.write("AFW"+ "\n")
#        f.write(str([x - fValOpt for x in fValAFW]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValAFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapAFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingAFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(activeSetAFW).replace("[", "").replace("]", "") + "\n")
#        #Output the FW Gap.
#        f.write("PFW"+ "\n")
#        f.write(str([x - fValOpt for x in fValPFW]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValPFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapPFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingPFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(activeSetPFW).replace("[", "").replace("]", "") + "\n")
#        #Output the FW Gap.
#        f.write("Lazy AFW"+ "\n")
#        f.write(str([x - fValOpt for x in fValAFWLazy]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write(str(activeSetLazy).replace("[", "").replace("]", "") + "\n")
#        f.write("LaCG-AFW"+ "\n")
#        f.write(str([x - fValOpt for x in fValLaCGAFW]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValLaCGAFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapLaCGAFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingLaCGAFW) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(restartsLaCG) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(activeLaCGAFW).replace("[", "").replace("]", "") + "\n")
#        f.write("LaCG-PFW"+ "\n")
#        f.write(str([x - fValOpt for x in fValLaCGPFW]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValLaCGPFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapLaCGPFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingLaCGPFW) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(restartsLaCGPFW) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(activeLaCGPFW).replace("[", "").replace("]", "") + "\n")
#        f.write("LaCG-AFW Lazy"+ "\n")
#        f.write(str([x - fValOpt for x in fValLaCGAFWLazy]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValLaCGAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapLaCGAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingLaCGAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write(str(restartsLaCGLazy) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(activeLaCGAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write("CGS"+ "\n")
#        f.write(str([x - fValOpt for x in fValCGS]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValCGS).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapCGS).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingCGS) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(iterationCGS) .replace("[", "").replace("]", "") + "\n")
#        f.write("Catalyst"+ "\n")
#        f.write(str([x - fValOpt for x in fCatalyst]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fCatalyst).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWCatalyst).replace("[", "").replace("]", "") + "\n")
#        f.write(str(tCatalyst) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(itCatalyst) .replace("[", "").replace("]", "") + "\n")


    """
    ----------------------------Traffic Congestion POLYTOPE--------------------------
    """
    
    TIME_LIMIT_REFERENCE_SOL = int(1800)
    TIME_LIMIT = int(800)
    MuVal = 0.1
    LVal = 10.0
    
    #Second feasible region.
    file = 'road_paths_01_DC_a.lp'
    tolerance = 2e-1
    
    pathfile = os.path.join(os.getcwd(), 'TrafficNetwork', file)
    feasibleRegion = GurobiPolytope(pathfile)
    size = feasibleRegion.dim()
    
    #Initial point
    x_0 = feasibleRegion.initialPoint()
    S_0 = [x_0]
    alpha_0 = [1]
    xOpt = np.random.rand(size)
    
    #Create objective function.
    fun = funcQuadraticDiag(size, xOpt, Mu = MuVal, L = LVal)
    
    #Decomposition Invariant CG
    print("Running algorithm to find optimum to high accuracy.")
    TIME_LIMIT = int(4*3600)
    print("Finding optimal solution to high accuracy using DIPFW or Lazy AFW.")
    xTest, FWGapTest, fValTest, timingTest, activeSetTest = runFW(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance/2.0, TIME_LIMIT_REFERENCE_SOL, FWVariant = "Lazy", typeStep = "EL", criterion = "DG")
    fValOpt = fValTest[-1]
    tolerance = min(np.asarray(FWGapTest))
    
    #Conditional Gradient Sliding.
    print("Running CGS.")
    CGS = CGS()
    xCGS, FWGapCGS, fValCGS, timingCGS, iterationCGS = CGS.run(x_0, fun, feasibleRegion, tolerance, TIME_LIMIT, criterion = "PG", criterionRef = fValOpt)
    
    #Catalyst augmented
    print("Running Catalyst-augmented AFW.")
    funCat = funcAccelSchemeDiag(len(x_0), fun.returnM(), fun.returnb(), fun.largestEig(), fun.smallestEig())
    CatalystAFW = catalystScheme()
    xCatalyst, FWCatalyst, fCatalyst, tCatalyst, itCatalyst = CatalystAFW.run(x_0, funCat, feasibleRegion, tolerance, TIME_LIMIT)
    
    #LaCG
    print("Running LaCG-AFW.")
    LaCGAway = LaCG()
    xLaCGAFW, FWGapLaCGAFW, fValLaCGAFW, timingLaCGAFW, activeLaCGAFW = LaCGAway.run(x_0, fun, feasibleRegion, tolerance, TIME_LIMIT, FWVariant = "AFW", typeStep = typeOfStep, criterionRef = fValOpt)
    restartsLaCG = LaCGAway.returnRestarts()
    restartsLaCG[:] = np.asarray([x - 1 for x in restartsLaCG])
    
    #LaCG Lazy
    print("Running LaCG-AFW Lazy.")
    LaCGAwayLazy = LaCG()
    xLaCGAFWLazy, FWGapLaCGAFWLazy, fValLaCGAFWLazy, timingLaCGAFWLazy, activeLaCGAFWLazy = LaCGAwayLazy.run(x_0, fun, feasibleRegion, tolerance, TIME_LIMIT, FWVariant = "Lazy", typeStep = typeOfStep, criterionRef = fValOpt)
    restartsLaCGLazy = LaCGAwayLazy.returnRestarts()
    restartsLaCGLazy[:] = np.asarray([x - 1 for x in restartsLaCGLazy])
    
    #LaCG PFW
    print("Running LaCG-PFW.")
    LaCGPairwise = LaCG()
    xLaCGPFW, FWGapLaCGPFW, fValLaCGPFW, timingLaCGPFW, activeLaCGPFW = LaCGPairwise.run(x_0, fun, feasibleRegion, tolerance, TIME_LIMIT, FWVariant = "PFW", typeStep = typeOfStep, criterionRef = fValOpt)
    restartsLaCGPFW = LaCGPairwise.returnRestarts()
    restartsLaCGPFW[:] = np.asarray([x - 1 for x in restartsLaCGPFW])
    
    #Vanilla PFW
    print("Running PFW.")
    xPFW, FWGapPFW, fValPFW, timingPFW, activeSetAFW = runFW(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance, TIME_LIMIT, FWVariant = "PFW", typeStep = typeOfStep, criterion = "PG", criterionRef = fValOpt)
    
    #Vanilla AFW
    print("Running AFW.")
    xAFW, FWGapAFW, fValAFW, timingAFW, activeSetPFW = runFW(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance, TIME_LIMIT, FWVariant = "AFW", typeStep = typeOfStep, criterion = "PG", criterionRef = fValOpt)
    
    #Run Lazy AFW
    print("Running Lazy AFW.")
    xAFWLazy, FWGapAFWLazy, fValAFWLazy, timingAFWLazy, activeSetLazy  = runFW(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance, TIME_LIMIT, FWVariant = "Lazy", typeStep = typeOfStep, criterion = "PG", criterionRef = fValOpt)
    
    #Plot primal gap in terms of iteration.
    plt.loglog(np.arange(len(fValAFW)), [(x - fValOpt) for x in fValAFW], '-*', color = 'b', markevery = np.logspace(0, np.log10(len(fValAFW)-1), 10).astype(int).tolist(), label = 'AFW')
    plt.loglog(np.arange(len(fValPFW)), [(x - fValOpt) for x in fValPFW], '-D', color = 'c',  markevery = np.logspace(0, np.log10(len(fValPFW)-1), 10).astype(int).tolist(), label = 'PFW')
    plt.loglog(np.arange(len(fValLaCGAFW)), [(x - fValOpt) for x in fValLaCGAFW], '-o', color = 'k', markevery = np.logspace(0, np.log10(len(fValLaCGAFW)-1), 10).astype(int).tolist(), label = 'LaCG-AFW')
    plt.loglog(np.arange(len(fValLaCGPFW)), [(x - fValOpt) for x in fValLaCGPFW], '-^', color = 'g', markevery = np.logspace(0, np.log10(len(fValLaCGPFW)-1), 10).astype(int).tolist(), label = 'LaCG-PFW')
    plt.loglog(iterationCGS, [(x - fValOpt) for x in fValCGS],  color = 'y',  label = 'CGS')
    plt.loglog(itCatalyst, [(x - fValOpt) for x in fCatalyst], ':', color = 'm',  label = 'Catalyst')
    plt.legend()
    plt.xlabel(r'$k$')
    plt.ylabel(r'$f(x_{k}) - f^{*}$')
    plt.grid()
    plt.show()
    plt.close()
    
    #Plot Primal gap in terms of time.
    plt.semilogy(timingAFW, [(x - fValOpt) for x in fValAFW], '-*', color = 'b', markevery = int(len(timingAFW)/10), label = 'AFW')
    plt.semilogy(timingPFW, [(x - fValOpt) for x in fValPFW], '-D',  color = 'g', markevery = int(len(timingPFW)/10), label = 'PFW')
    plt.semilogy(timingLaCGAFW, [(x - fValOpt) for x in fValLaCGAFW], '-o', color = 'k', markevery = int(len(timingLaCGAFW)/10), label = 'LaCG-AFW')
    plt.semilogy(timingLaCGPFW, [(x - fValOpt) for x in fValLaCGPFW], '-^',  color = 'g', markevery = int(len(timingLaCGPFW)/10), label = 'LaCG-PFW')
    plt.semilogy(timingCGS, [(x - fValOpt) for x in fValCGS],   color = 'y', label = 'CGS')
    plt.semilogy(tCatalyst, [(x - fValOpt) for x in fCatalyst], ':',  color = 'm', label = 'Catalyst')
    plt.legend()
    plt.ylabel(r'$f(x_{k}) - f^{*}$')
    plt.xlabel(r't[s]')
    plt.grid()
    plt.show()
    plt.close()
    
#    #Generate a timestamp for the example and save data
#    ts = time.time()
#    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S').replace(' ', '-').replace(':', '-')
#    with open(os.path.join(os.getcwd(),  "VideoColocalization_" + str(timestamp) + "_Mu" + str(MuVal) + "_L" + str(LVal)+ "_Size" + str(size)  + "_TypeStep_" + typeOfStep  + ".txt"), 'w') as f:
#        f.write("size:\t" + str(len(x_0)) + "\n")
#        f.write("Mu:\t" + str(fun.smallestEig())+ "\n")
#        f.write("LVal:\t" + str(fun.largestEig())+ "\n")
#        f.write("Tolerance:\t" + str(tolerance)+ "\n")
#        f.write("Optimum:\t" + str(fValOpt)+ "\n")
#        #Output the FW Gap.
#        f.write("AFW"+ "\n")
#        f.write(str([x - fValOpt for x in fValAFW]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValAFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapAFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingAFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(activeSetAFW).replace("[", "").replace("]", "") + "\n")
#        #Output the FW Gap.
#        f.write("PFW"+ "\n")
#        f.write(str([x - fValOpt for x in fValPFW]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValPFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapPFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingPFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(activeSetPFW).replace("[", "").replace("]", "") + "\n")
#        #Output the FW Gap.
#        f.write("Lazy AFW"+ "\n")
#        f.write(str([x - fValOpt for x in fValAFWLazy]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write(str(activeSetLazy).replace("[", "").replace("]", "") + "\n")
#        f.write("LaCG-AFW"+ "\n")
#        f.write(str([x - fValOpt for x in fValLaCGAFW]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValLaCGAFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapLaCGAFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingLaCGAFW) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(restartsLaCG) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(activeLaCGAFW).replace("[", "").replace("]", "") + "\n")
#        f.write("LaCG-PFW"+ "\n")
#        f.write(str([x - fValOpt for x in fValLaCGPFW]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValLaCGPFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapLaCGPFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingLaCGPFW) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(restartsLaCGPFW) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(activeLaCGPFW).replace("[", "").replace("]", "") + "\n")
#        f.write("LaCG-AFW Lazy"+ "\n")
#        f.write(str([x - fValOpt for x in fValLaCGAFWLazy]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValLaCGAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapLaCGAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingLaCGAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write(str(restartsLaCGLazy) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(activeLaCGAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write("CGS"+ "\n")
#        f.write(str([x - fValOpt for x in fValCGS]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValCGS).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapCGS).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingCGS) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(iterationCGS) .replace("[", "").replace("]", "") + "\n")
#        f.write("Catalyst"+ "\n")
#        f.write(str([x - fValOpt for x in fCatalyst]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fCatalyst).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWCatalyst).replace("[", "").replace("]", "") + "\n")
#        f.write(str(tCatalyst) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(itCatalyst) .replace("[", "").replace("]", "") + "\n")

    """
    ------------------------------VideoColocalization POLYTOPE----------------------------
    """
    from feasibleRegions import flowPolytope
    
    TIME_LIMIT_REFERENCE_SOL = int(1800)
    TIME_LIMIT = int(1800)
    MuVal = 0.1
    LVal = 10.0
    tolerance = 1.0e-4
    typeOfStep = "SS"
    
    feasibleRegion = flowPolytope(15, 15, typeGraph = "Structured")
    
    size = feasibleRegion.dim()
    
    #Initial point
    x_0 = feasibleRegion.initialPoint()
    S_0 = [x_0]
    alpha_0 = [1]
    xOpt = np.zeros(size)
    
    matrix = randomPSDGeneratorSparse(size, 0.01)
    LVal, MuVal = calculateEigenvalues(matrix)
    
    """# CREATE MATRIX AND EXPORT."""
    #Create a matrix and export it.
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S').replace(' ', '-').replace(':', '-')
    fun = funcQuadratic(size, csc_matrix(matrix), xOpt, Mu = MuVal, L = LVal)
    
    
    """# SOLVE AND EXPORT THE SOLUTION."""
    #Decomposition Invariant CG
    print("Running algorithm to find optimum to high accuracy.")
    TIME_LIMIT = int(1800)
    xTest, FWGapTest, fValTest, timingTest = DIPFW(x_0, fun, feasibleRegion, tolerance/2.0, TIME_LIMIT_REFERENCE_SOL, typeStep = "EL", criterion = "DG")
    fValOpt = fValTest[-1]
    tolerance = min(np.asarray(FWGapTest))

    TIME_LIMIT = int(1200)
    
    #Conditional Gradient Sliding.
    print("Running CGS.")
    CGS = CGS()
    xCGS, FWGapCGS, fValCGS, timingCGS, iterationCGS = CGS.run(x_0, fun, feasibleRegion, tolerance, TIME_LIMIT, criterion = "PG", criterionRef = fValOpt)
    
    #Catalyst augmented
    print("Running Catalyst-augmented AFW.")
    funCat = funcAccelScheme(len(x_0), matrix, fun.returnb(), fun.largestEig(), fun.smallestEig())
    CatalystAFW = catalystScheme()
    xCatalyst, FWCatalyst, fCatalyst, tCatalyst, itCatalyst = CatalystAFW.run(x_0, funCat, feasibleRegion, tolerance, TIME_LIMIT)
    
    #Decomposition Invariant CG
    print("\nRunning DICG.")
    xDICG, FWGapDICG, fValDICG, timingDICG = DIPFW(x_0, fun, feasibleRegion, tolerance, TIME_LIMIT, typeStep = typeOfStep, criterion = "PG", criterionRef = fValOpt)
    
    #LaCG
    print("Running LaCG-AFW.")
    LaCGAway = LaCG()
    xLaCGAFW, FWGapLaCGAFW, fValLaCGAFW, timingLaCGAFW, activeLaCGAFW = LaCGAway.run(x_0, fun, feasibleRegion, tolerance, TIME_LIMIT, FWVariant = "AFW", typeStep = typeOfStep, criterionRef = fValOpt)
    restartsLaCG = LaCGAway.returnRestarts()
    restartsLaCG[:] = np.asarray([x - 1 for x in restartsLaCG])
    
    #LaCG Lazy
    print("Running LaCG-AFW Lazy.")
    LaCGAwayLazy = LaCG()
    xLaCGAFWLazy, FWGapLaCGAFWLazy, fValLaCGAFWLazy, timingLaCGAFWLazy, activeLaCGAFWLazy = LaCGAwayLazy.run(x_0, fun, feasibleRegion, tolerance, TIME_LIMIT, FWVariant = "Lazy", typeStep = typeOfStep, criterionRef = fValOpt)
    restartsLaCGLazy = LaCGAwayLazy.returnRestarts()
    restartsLaCGLazy[:] = np.asarray([x - 1 for x in restartsLaCGLazy])
    
    #LaCG PFW
    print("Running LaCG-PFW.")
    LaCGPairwise = LaCG()
    xLaCGPFW, FWGapLaCGPFW, fValLaCGPFW, timingLaCGPFW, activeLaCGPFW = LaCGPairwise.run(x_0, fun, feasibleRegion, tolerance, TIME_LIMIT, FWVariant = "PFW", typeStep = typeOfStep, criterionRef = fValOpt)
    restartsLaCGPFW = LaCGPairwise.returnRestarts()
    restartsLaCGPFW[:] = np.asarray([x - 1 for x in restartsLaCGPFW])
    
    #Vanilla PFW
    print("Running PFW.")
    xPFW, FWGapPFW, fValPFW, timingPFW, activeSetAFW = runFW(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance, TIME_LIMIT, FWVariant = "PFW", typeStep = typeOfStep, criterion = "PG", criterionRef = fValOpt)
    
    #Vanilla AFW
    print("Running AFW.")
    xAFW, FWGapAFW, fValAFW, timingAFW, activeSetPFW = runFW(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance, TIME_LIMIT, FWVariant = "AFW", typeStep = typeOfStep, criterion = "PG", criterionRef = fValOpt)
    
    #Run Lazy AFW
    print("Running Lazy AFW.")
    xAFWLazy, FWGapAFWLazy, fValAFWLazy, timingAFWLazy, activeSetLazy  = runFW(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance, TIME_LIMIT, FWVariant = "Lazy", typeStep = typeOfStep, criterion = "PG", criterionRef = fValOpt)
    
    #Plot primal gap in terms of iteration.
    plt.loglog(np.arange(len(fValAFW)), [(x - fValOpt) for x in fValAFW], '-*', color = 'b', markevery = np.logspace(0, np.log10(len(fValAFW)-1), 10).astype(int).tolist(), label = 'AFW')
    plt.loglog(np.arange(len(fValPFW)), [(x - fValOpt) for x in fValPFW], '-D', color = 'c',  markevery = np.logspace(0, np.log10(len(fValPFW)-1), 10).astype(int).tolist(), label = 'PFW')
    plt.loglog(np.arange(len(fValAFWLazy)), [(x - fValOpt) for x in fValAFWLazy], '--*', color = 'b', markerfacecolor='none', markevery = np.logspace(0, np.log10(len(fValAFWLazy)-1), 10).astype(int).tolist(), label = 'AFW (L)')
    plt.loglog(np.arange(len(fValLaCGAFW)), [(x - fValOpt) for x in fValLaCGAFW], '-o', color = 'k', markevery = np.logspace(0, np.log10(len(fValLaCGAFW)-1), 10).astype(int).tolist(), label = 'LaCG-AFW')
    plt.loglog(np.arange(len(fValLaCGPFW)), [(x - fValOpt) for x in fValLaCGPFW], '-^', color = 'g', markevery = np.logspace(0, np.log10(len(fValLaCGPFW)-1), 10).astype(int).tolist(), label = 'LaCG-PFW')
    plt.loglog(np.arange(len(fValLaCGAFWLazy)), [(x - fValOpt) for x in fValLaCGAFWLazy], '--o', color = 'k', markerfacecolor='none', markevery = np.logspace(0, np.log10(len(fValLaCGAFWLazy)-1), 10).astype(int).tolist(), label = 'LaCG-AFW (L)')
    plt.loglog(np.arange(len(fValDICG)), [(x - fValOpt) for x in fValDICG], '-s', color = 'r', markerfacecolor='none', markevery = np.logspace(0, np.log10(len(fValDICG)-1), 10).astype(int).tolist(), label = 'DICG')
    plt.loglog(iterationCGS, [(x - fValOpt) for x in fValCGS],  color = 'y',  label = 'CGS')
    plt.loglog(itCatalyst, [(x - fValOpt) for x in fCatalyst], ':', color = 'm',  label = 'Catalyst')
    plt.legend()
    plt.xlabel(r'$k$')
    plt.ylabel(r'$f(x_{k}) - f^{*}$')
    plt.grid()
    plt.show()
    plt.close()
    
    #Plot Primal gap in terms of time.
    plt.semilogy(timingAFW, [(x - fValOpt) for x in fValAFW], '-*', color = 'b', markevery = int(len(timingAFW)/10), label = 'AFW')
    plt.semilogy(timingPFW, [(x - fValOpt) for x in fValPFW], '-D',  color = 'g', markevery = int(len(timingPFW)/10), label = 'PFW')
    plt.semilogy(timingLaCGAFW, [(x - fValOpt) for x in fValLaCGAFW], '-o', color = 'k', markevery = int(len(timingLaCGAFW)/10), label = 'LaCG-AFW')
    plt.semilogy(timingLaCGPFW, [(x - fValOpt) for x in fValLaCGPFW], '-^',  color = 'g', markevery = int(len(timingLaCGPFW)/10), label = 'LaCG-PFW')
    plt.semilogy(timingDICG, [(x - fValOpt) for x in fValDICG], '-s',  color = 'r', markevery = int(len(timingDICG)/10), label = 'DICG')
    plt.semilogy(timingCGS, [(x - fValOpt) for x in fValCGS],   color = 'y', label = 'CGS')
    plt.semilogy(tCatalyst, [(x - fValOpt) for x in fCatalyst], ':',  color = 'm', label = 'Catalyst')
    plt.legend()
    plt.ylabel(r'$f(x_{k}) - f^{*}$')
    plt.xlabel(r't[s]')
    plt.grid()
    plt.show()
    plt.close()
        
#    #Generate a timestamp for the example and save data
#    ts = time.time()
#    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S').replace(' ', '-').replace(':', '-')
#    with open(os.path.join(os.getcwd(),  "VideoColocalization_" + str(timestamp) + "_Mu" + str(MuVal) + "_L" + str(LVal)+ "_Size" + str(size)  + "_TypeStep_" + typeOfStep  + ".txt"), 'w') as f:
#        f.write("size:\t" + str(len(x_0)) + "\n")
#        f.write("Mu:\t" + str(fun.smallestEig())+ "\n")
#        f.write("LVal:\t" + str(fun.largestEig())+ "\n")
#        f.write("Tolerance:\t" + str(tolerance)+ "\n")
#        f.write("Optimum:\t" + str(fValOpt)+ "\n")
#        #Output the FW Gap.
#        f.write("AFW"+ "\n")
#        f.write(str([x - fValOpt for x in fValAFW]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValAFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapAFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingAFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(activeSetAFW).replace("[", "").replace("]", "") + "\n")
#        #Output the FW Gap.
#        f.write("PFW"+ "\n")
#        f.write(str([x - fValOpt for x in fValPFW]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValPFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapPFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingPFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(activeSetPFW).replace("[", "").replace("]", "") + "\n")
#        #Output the FW Gap.
#        f.write("Lazy AFW"+ "\n")
#        f.write(str([x - fValOpt for x in fValAFWLazy]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write(str(activeSetLazy).replace("[", "").replace("]", "") + "\n")
#        f.write("LaCG-AFW"+ "\n")
#        f.write(str([x - fValOpt for x in fValLaCGAFW]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValLaCGAFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapLaCGAFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingLaCGAFW) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(restartsLaCG) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(activeLaCGAFW).replace("[", "").replace("]", "") + "\n")
#        f.write("LaCG-PFW"+ "\n")
#        f.write(str([x - fValOpt for x in fValLaCGPFW]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValLaCGPFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapLaCGPFW).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingLaCGPFW) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(restartsLaCGPFW) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(activeLaCGPFW).replace("[", "").replace("]", "") + "\n")
#        f.write("LaCG-AFW Lazy"+ "\n")
#        f.write(str([x - fValOpt for x in fValLaCGAFWLazy]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValLaCGAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapLaCGAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingLaCGAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write(str(restartsLaCGLazy) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(activeLaCGAFWLazy).replace("[", "").replace("]", "") + "\n")
#        f.write("CGS"+ "\n")
#        f.write(str([x - fValOpt for x in fValCGS]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValCGS).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapCGS).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingCGS) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(iterationCGS) .replace("[", "").replace("]", "") + "\n")
#        f.write("DICG"+ "\n")
#        f.write(str([x - fValOpt for x in fValDICG]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fValDICG).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWGapDICG).replace("[", "").replace("]", "") + "\n")
#        f.write(str(timingDICG) .replace("[", "").replace("]", "") + "\n")
#        f.write("Catalyst"+ "\n")
#        f.write(str([x - fValOpt for x in fCatalyst]).replace("[", "").replace("]", "") + "\n")
#        f.write(str(fCatalyst).replace("[", "").replace("]", "") + "\n")
#        f.write(str(FWCatalyst).replace("[", "").replace("]", "") + "\n")
#        f.write(str(tCatalyst) .replace("[", "").replace("]", "") + "\n")
#        f.write(str(itCatalyst) .replace("[", "").replace("]", "") + "\n")
