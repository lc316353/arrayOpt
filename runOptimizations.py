# -*- coding: utf-8 -*-

import arrayOpt as AO
import numpy as np
from sys import argv


ID = 0 #int(argv[1])

multiple = [1,2,3,5,10]

tag = "multiple"+str(multiple[ID])

if __name__=="__main__":
    
    N = 20
    freq = 10
    SNR = 15
    p = 0.2
    loss = "mean"

    if multiple[ID]==1:
        ar = AO.AnalyticResidual(default_mode="volume", e1=AO.e1_sym, e2=AO.e2_sym)
    else:
        ar = AO.AnalyticResidual(default_mode="volume multiple"+str(multiple[ID]), e1=AO.e1_sym, e2=AO.e2_sym)

    chain = [
        ("PSO",   {}),
        ("Adam", {}),
    ]

    final_res, final_pos = ar.optimize_chain(N, freq, SNR, p, chain, loss=loss, savename=str(N)+tag+"Resultsall"+str(N))

    print(f"  Final residual : {final_res:.6f}")
    print(f"  Final position : {np.array(final_pos).round(2)}")

