# -*- coding: utf-8 -*-

import arrayOpt as AO
import numpy as np
from sys import argv


ID = 1 #int(argv[1])

multiple = [1,2,3,5,10]

tag = "multiple"+str(multiple[ID])

if __name__=="__main__":
    
    N = 3
    freq = [3,6,10] 
    SNR = 15
    p = 0.2
    mirror = "mean"
    residual_limits = 1/np.array([1.1,1.1,1.07])
    cost_limit = 1
    loss = "broadband"

    if multiple[ID]==1:
        ar = AO.AnalyticResidual(default_mode="volume", e1=AO.e1_sym, e2=AO.e2_sym)
    else:
        ar = AO.AnalyticResidual(default_mode="volume multiple"+str(multiple[ID]), e1=AO.e1_sym, e2=AO.e2_sym)

    chain = [
        ("PSO",   {}),
        ("Adam", {}),
    ]

    final_res, final_pos = ar.optimize_chain(N, freq, SNR, p, chain, mirror=mirror, loss=loss, cost_limit=cost_limit, residual_limits=residual_limits, savename=str(N)+tag+"Resultsall"+str(N))

    print(f"  Final loss : {final_res:.6f}")
    print(f"  Final position : {np.array(final_pos).round(2)}")

    freqs=np.linspace(1,10,30)
    resids=[]
    for fff in freqs:
        resids.append(ar.residual(final_pos,N,fff,SNR,p,"max"))
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(freqs, 1/np.array(resids))
    plt.scatter(freq, 1/residual_limits, marker="s",color="tab:green")
    plt.xlabel(r"Test frequency $f$ [Hz]")
    plt.ylabel(r"Mitigation factor $M$")
    plt.grid()

    fig, ax = ar.new_state_plot_3D(mirrorcolor="blue")
    ar.plot_state_3D(ax, final_pos, N, color="k", marker="d")

"""
#From this state foreward, Adam fails -> investigate
PSO_state=[  14.61708993 , 598.48739024 ,-265.07142721 ,-372.06285598 ,-518.59721515
,-284.70204736 ,-184.93362106 , 131.90743272,  109.68426772  ,623.86862721
  ,791.77413673 , 137.77710893 , 834.02884536, -841.99414578 ,-137.05883428
 ,-218.95410846 ,-522.79469517 , 284.00702288, -762.19241654 ,-428.21143776
  , 29.03642203 , 836.49079465 , 174.70422906, -173.90309634 , 906.42735737
 ,-760.78694806 , 275.54437674 , 474.802897   , 234.63417824 , 290.99795666]

ar.loss_function(PSO_state, N, freq, SNR, p, loss=loss, cost_limit=cost_limit, residual_limits=residual_limits)

ttstate=np.array([  87.26648358,  661.288157  , -277.98318188, -412.9873268 ,
-472.15759804, -301.7616351 , -257.27289315,  198.09598259,
229.02649002,  687.30921396,  863.77255332,   61.92053046,
787.55504761, -901.56365615, -217.44807179, -284.89956635,
-591.99598205,  298.47108503, -846.81772505, -340.92236482,
 -6.87638039,  922.80287559,  248.49122236, -292.37477695,
974.56969644, -694.67107954,  274.36954528,  564.78411459,
328.97559337,  294.84483866])

trstate=np.array([  84.63100328,  659.09720572, -277.98302559, -410.23094465,
-469.50362476, -300.37099742, -254.87568034,  195.37127112,
 226.07187275,  689.42783666,  860.84469292,   63.86984752,
 786.06938551, -898.53104979, -214.47761215, -287.45013885,
-589.0500325 ,  298.02779345, -844.5821659 , -343.51419678,
  -7.99440122,  922.6316969 ,  246.20077055, -289.75669836,
 976.0777227 , -692.99453861,  274.33904239,  562.67820752,
 325.8742556 ,  294.8448314 ])

"""