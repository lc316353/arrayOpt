# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:06:38 2026

@author: schillings
"""

import arrayOpt as AO
import numpy as np
import matplotlib.pyplot as plt
import os


plt.rc('legend',fontsize=22,title_fontsize=22)
plt.rc('axes',labelsize=25,titlesize=22)
plt.rc("xtick",labelsize=20)
plt.rc("ytick",labelsize=20)
plt.rc('figure',figsize=(10,9))
plt.rc('font',size=30)
plt.close("all")

if __name__ == '__main__':
    ar = AO.AnalyticResidual(e1=AO.e1_sym, e2=AO.e2_sym)       
    
    
    #~Parameters~#
    path = "broadband/"
    
    freqs=np.linspace(1,10,50)
    
    N = 20          #number of boreholes
    multi = 1       #seismometers per borehole
    Ntunnel = 50    #seismometers per tunnel
    displace = 0    #m random displacement from optimal position
    loss = "max"
    mode = "disallB" #"Nplot", "disallB", ""
    #"Nplot" produces curves for different numbers of boreholes in one plot
    #"disallB" and "" produces curves for different numbers of seismometers per borehole
    #"disallB" displaces boreholes and z-positions of each borehole randomly
    #"" only displaces boreholes
    #Scroll down to TODO if you change more from or to "Nplot"
    
    if displace>0:
        statistics = 10
    else:
        statistics = 1
    
    
    #~Broadband plot~#
    fig, ax = plt.subplots() 
    
    plt.xlabel(r"Test frequency $f$ [Hz]")
    plt.xlim(1,10)
    
    plt.ylabel(r"Mitigation factor $M$")
    if mode=="Nplot":
        plt.ylim(1,25)
    elif N==50:
        plt.ylim(1,65)
    else:
        plt.ylim(1,25)
    
    #Plot description
    if mode != "Nplot":
        plt.plot([], [], "w-", label=r"optimize "+str(N)+" boreholes per corner at 10 Hz")
    else: 
        plt.plot([], [], "w-", label=r"optimize $N$"+" boreholes per corner at 10 Hz")
    plt.plot([], [], "w-", label="add "+str(Ntunnel*2)+" seismometers per tunnel")
    if statistics > 1:
        plt.plot([], [], "w-", label=r"vary Gaussian displacement with $\sigma=$"+str(displace)+" m")
    else:
        plt.plot([], [], "w-", label=r"no Gaussian displacement")
    j = len(ax.get_legend_handles_labels()[0])
    
    linestyles=["solid", "dashed", "dashdot", "dotted", (0, (3, 3, 1, 3, 1, 3)), (0, (3, 3, 1, 3, 3, 3,))]
    
    
    #for i, N in enumerate([20,30,40,50,60,70]):    # TODO: use this if mode=="Nplot"
    for i, multi in enumerate([1,2,3,5,10]):           # TODO: use this if mode!="Nplot"
        
        all_resids = np.zeros((len(freqs), statistics))
        
        for seed in range(max(1, statistics)):
            np.random.seed(seed)
            
            ar.set_default_mode("volume multiple"+str(multi))
            
            #read data file
            if mode!="Nplot":
                data = AO.ReadData("multiple"+str(multi), "resultFiles", N, i=1)
            else:
                data = AO.ReadData("PSOmeanlBf10polished", "resultFiles", N)
            
            newstate = data.state
            newstate = newstate.reshape((N,multi+2))
            #"volume multipleX" has X+2 z-coordinates for each borehole
            
            #per borehole displacement
            if mode=="":
                newstate += np.array([np.random.normal(0,displace,N),np.random.normal(0,displace,N)]+[[0]*N]*multi).T
            elif mode=="disallB":
                newstate+= np.random.normal(0,displace,newstate.shape)
            
            newstate = ar.multiple_to_volume_state(newstate, multi)
            #"volume" counts all N*multi seismometers independently
            
            #volume displacement
            if mode=="Nplot":
                newstate+= np.random.normal(0,displace,newstate.shape)
            newstate = np.concatenate((newstate, np.array([np.linspace(0,5000,Ntunnel)*ar.e1[0],np.linspace(0,5000,Ntunnel)*ar.e1[1],np.zeros(Ntunnel)]).T, np.array([np.linspace(0,5000,Ntunnel)*ar.e2[0],np.linspace(0,5000,Ntunnel)*ar.e2[1],np.zeros(Ntunnel)]).T))
            
            ar.set_default_mode("volume")
            
            #precalculation and saving of broadband residuals
            if mode=="Nplot":
                filename="N"+str(N)+"t"+str(Ntunnel)+"dis"+str(displace)+str(loss)+str(seed)+".npy"
            elif mode=="disallB" and statistics>1:
                filename="N"+str(N)+"multi"+str(multi)+"t"+str(Ntunnel)+"disAllB"+str(displace)+str(loss)+str(seed)+".npy"
            else:
                filename="N"+str(N)+"multi"+str(multi)+"t"+str(Ntunnel)+"dis"+str(displace)+str(loss)+str(seed)+".npy"
            
            if not os.path.exists(path+filename):
                
                resids=[]
                for freq in freqs:
                    resids.append(ar.residual(newstate,2*Ntunnel+data.N*multi,freq,data.SNR,data.p,"max"))
                np.save(path+filename,resids)
            all_resids[:,seed] = np.load(path+filename)
            
        #plotting
        if statistics>2:
            #plt.fill_between(freqs,1/np.max(all_resids,axis=1),1/np.min(all_resids,axis=1), alpha=0.3)
            plt.fill_between(freqs, 1/(np.mean(all_resids,axis=1)+np.std(all_resids,axis=1)), 1/(np.mean(all_resids,axis=1)-np.std(all_resids,axis=1)), alpha=0.3)
        plt.plot(freqs, 1/np.mean(all_resids,axis=1), label=str(multi)+" ("+str(multi*N+2*Ntunnel)+")", linestyle=linestyles[i], linewidth=2)
    
    #legend
    h, l = ax.get_legend_handles_labels()
    leg = plt.legend(h[:j],l[:j],loc="lower center", bbox_to_anchor=(0.5,0.885), handlelength=0)
    ax.add_artist(leg)
    leg2 = plt.legend(h[j:],l[j:],loc="upper left",bbox_to_anchor=(0,0.915), title="seismometers\nper borehole (total)")
    leg2._legend_box.align = "left"
    
    plt.grid()
    
    #saving
    if mode=="Nplot":
        plt.savefig(path+"broadbandNPlot"+"t"+str(Ntunnel)+"dis"+str(displace)+str(loss)+".pdf")
    elif mode=="disallB" and statistics>1:
        plt.savefig(path+"broadbandPlotN"+str(N)+"t"+str(Ntunnel)+"disAllB"+str(displace)+str(loss)+".pdf")
    else:
        plt.savefig(path+"broadbandPlotN"+str(N)+"t"+str(Ntunnel)+"dis"+str(displace)+str(loss)+".pdf")
    
    
    #~3D-Plot of geometry and state~#
    fig, ax = ar.new_state_plot_3D(mirrorcolor="tab:blue", mirrormarkersize=50)
    
    ar.plot_state_3D(ax, newstate, 2*Ntunnel+data.N*multi, marker="d", color="white")
    ar.plot_state_3D(ax, newstate, 2*Ntunnel+data.N*multi, marker="d", color="black", markersize=11)
    
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    ax.zaxis.labelpad = 20
    
    plt.savefig("geometry.svg")  
    
    
    #~2D topview of borehole distribution~#
    fig, ax = plt.subplots()
    
    state = data.state.reshape((N,multi+2))
    tunnelseismometers = np.concatenate((np.array([np.linspace(0,5000,Ntunnel)*ar.e1[0],np.linspace(0,5000,Ntunnel)*ar.e1[1],np.zeros(Ntunnel)]).T, np.array([np.linspace(0,5000,Ntunnel)*ar.e2[0],np.linspace(0,5000,Ntunnel)*ar.e2[1],np.zeros(Ntunnel)]).T))
    
    mirror_x = np.array([ar.d_in1*ar.e1[0], ar.d_end1*ar.e1[0], ar.d_in2*ar.e2[0], ar.d_end2*ar.e2[0]])
    mirror_y = np.array([ar.d_in1*ar.e1[1], ar.d_end1*ar.e1[1], ar.d_in2*ar.e2[1], ar.d_end2*ar.e2[1]])

    plt.xlabel(r"$x$ [m]")
    plt.xlim(-500, 1000)
    plt.ylabel(r"$y$ [m]")
    plt.ylim(-750, 750)
    
    plt.scatter(tunnelseismometers[:,0], tunnelseismometers[:,1], c="gray", s=350, marker="d")
        
    plt.scatter(mirror_x, mirror_y, c="k", s=350, marker="o")
    plt.scatter(mirror_x, mirror_y, c="blue", s=250, marker="o")
    
    plt.scatter(state[:,0], state[:,1], c="k", s=450, marker="d")
    plt.scatter(state[:,0], state[:,1], c="white", s=350, marker="d")
        
    plt.grid()

    plt.savefig("presentablePositionPlot.svg")
         