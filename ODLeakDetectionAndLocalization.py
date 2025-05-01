# -*- coding: utf-8 -*-
from __future__ import print_function
import wntr
import numpy as np
import os
#import csv
import pandas as pd#(matheus)import pandas
import time
import matplotlib.pyplot as plt
from wntr.epanet.util import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#import sys

# Set duration in hours
sim_step_minutes = 30
durationHours = 24*365 # One Year
timeStamp = pd.date_range("2024-01-01 00:00", "2024-12-30 23:55", freq=str(sim_step_minutes)+"min")

DATASET_FOLDER_NAME = "EPANET Net 3_OD"
WDSName = "EPANET Net 3"

df_results = pd.read_csv(DATASET_FOLDER_NAME+'/Summary.csv', header=0, index_col=0)

print(["Run input file: ", DATASET_FOLDER_NAME])

INP_UNITS = FlowUnits.LPS

dataset_folder_os = os.getcwd()+'\\'+DATASET_FOLDER_NAME

plot_graphs=False
# plot_graphs=True

results_decimal_digits = 5

# RUN SCENARIOS
def runScenarios(scNum):
    print('Running scenario: '+str(scNum))
    
    #Define paths
    sc_folder_os = dataset_folder_os+'\\Scenario-'+str(scNum)

    OD_folder_wntr = DATASET_FOLDER_NAME + '/Scenario-'+str(scNum)
    OD_inp_file_wntr = OD_folder_wntr + '/' + WDSName +'_OnlyDemands-'+str(scNum) + '.inp'

    ODdem = pd.read_csv(OD_folder_wntr +'/OD_Node_demands.csv', index_col=[0])
    ODdem.index = pd.to_datetime(ODdem.index)

    ODest_node_press = pd.read_csv(OD_folder_wntr +'/OD_Estimated_Node pressures.csv', index_col=[0])
    ODest_node_press.index = pd.to_datetime(ODest_node_press.index)

    ODest_link_flows = pd.read_csv(OD_folder_wntr +'/OD_Estimated_Link_flows.csv', index_col=[0])
    ODest_link_flows.index = pd.to_datetime(ODest_link_flows.index)

    real_leak_demand0 = pd.read_csv(OD_folder_wntr +'/Leaks/Leak_leak_node0_demand.csv', index_col=[0])
    real_leak_demand0.index = pd.to_datetime(real_leak_demand0.index)

    wn = wntr.network.WaterNetworkModel(OD_inp_file_wntr)

    #Calculate loss signal
    loss = - ODdem.sum(axis=1)

    #Graph plots for paper
    if scNum==88 and plot_graphs==True:
        real_leak_demand0 = pd.read_csv(OD_folder_wntr +'/Leaks/Leak_leak_node0_demand.csv', index_col=[0])
        real_leak_demand0.index = pd.to_datetime(real_leak_demand0.index)

        fig = plt.figure(1)
        ax = fig.add_subplot(3,1,1)
        ax.set_title("Scenario 88: 1 leak node, 1 unmeasured node,\n 1 fraud node, 0% demand measurement error\n(a) Water loss and real leak demand")
        ax.set_ylabel("Water Loss/ \n Leak Demand (L/s)")
        xvals = np.arange(len(loss))*0.5
        ax.plot(xvals, loss, color='g', label='Measured \n  water Loss')
        ax.plot(xvals,real_leak_demand0, color='r', label='Real \n  leak demand')
        ax.legend(loc='upper left')
        
        ax = fig.add_subplot(3,1,2)
        ax.set_title("(b) Water loss and real leak demand (zoom)")
        ax.set_ylabel("Water Loss/ \n Real leak Demand (L/s)")
        ax.plot(xvals, loss, color='g', label='Measured \n  water Loss')
        ax.plot(xvals,real_leak_demand0, color='r', label='Real \n  leak demand')
        ax.axis([6800, 7050, 7, 12])
        ax.legend(loc='upper left')

        ax = fig.add_subplot(3,1,3)
        ax.set_title("(c) Estimated node pressures")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Pressure (m)")
        ax.plot(xvals, ODest_node_press.loc[:,'115'], color='black', label='Node 115')
        ax.plot(xvals, ODest_node_press.loc[:,'215'], color='blue', label='Node 215')
        ax.plot(xvals, ODest_node_press.loc[:,'141'], color='brown', label='Node 141')
        ax.axis([6800, 7050, 35, 60])
        ax.legend(loc='upper left')
        plt.show()

        node_press = pd.read_csv(OD_folder_wntr +'/Node_pressures.csv', index_col=[0])
        node_press.index = pd.to_datetime(node_press.index)

        fig = plt.figure(2)
        ax = fig.add_subplot(2,1,1)
        ax.set_title("Scenario 88: Difference between real \n and estimated pressures\n (a) Node 115, near leakage")
        ax.set_ylabel("Pressure (m)")
        ax.plot(xvals, node_press.loc[:,'115'], color='blue', label='Node 115 - Real')
        ax.plot(xvals, ODest_node_press.loc[:,'115'], color='red', label='Node 115 - Est.', linestyle='dashed')
        ax.axis([6850, 7050, 35, 50])
        ax.legend(loc='upper left')

        ax = fig.add_subplot(2,1,2)
        ax.set_title("(b) Node 215, far from leakage")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Pressure (m)")
        ax.plot(xvals, node_press.loc[:,'215'], color='blue', label='Node 215 - Real')
        ax.plot(xvals, ODest_node_press.loc[:,'215'], color='red', label='Node 215 - Est.', linestyle='dashed')
        ax.axis([6850, 7050, 35, 50])
        ax.legend(loc='upper left')
        plt.show()

        link_flows = pd.read_csv(OD_folder_wntr +'/Link_flows.csv', index_col=[0])
        link_flows.index = pd.to_datetime(link_flows.index)

        fig = plt.figure(3)
        ax = fig.add_subplot(3,1,1)
        ax.set_title("Scenario 88: Difference between real \n and estimated link flows\n (a) Pump 335")
        ax.set_ylabel("Flow (L/s)")
        ax.plot(xvals, link_flows.loc[:,'335'], color='blue', label='Pipe 335 - Real')
        ax.plot(xvals, ODest_link_flows.loc[:,'335'], color='red', label='Pipe 335 - Est', linestyle='dashed')
        ax.axis([6820, 7000, -5, 950])
        ax.legend(loc='lower left')
        
        ax = fig.add_subplot(3,1,2)
        ax.set_title("(b) Pipe 112 - leaky pipe")
        ax.set_ylabel("Flow (L/s)")
        ax.plot(xvals, link_flows.loc[:,'112'], color='blue', label='Pipe 122 - Real')
        ax.plot(xvals, ODest_link_flows.loc[:,'112'], color='red', label='Pipe 122 - Est', linestyle='dashed')
        ax.axis([6820, 7000, -5, 25])
        ax.legend(loc='upper left')

        ax = fig.add_subplot(3,1,3)
        ax.set_title("(b) Pipe 40 - Tank 1 output")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Flow (L/s)")
        ax.plot(xvals, link_flows.loc[:,'40'], color='blue', label='Pipe 40 - Real')
        ax.plot(xvals, ODest_link_flows.loc[:,'40'], color='red', label='Pipe 40 - Est', linestyle='dashed')
        ax.axis([6820, 7000, -200, 50])
        ax.legend(loc='lower left')

        plt.show()

    if scNum==324 and plot_graphs==True:
        real_leak_demand0 = pd.read_csv(OD_folder_wntr +'/Leaks/Leak_leak_node0_demand.csv', index_col=[0])
        real_leak_demand0.index = pd.to_datetime(real_leak_demand0.index)

        real_leak_demand1 = pd.read_csv(OD_folder_wntr +'/Leaks/Leak_leak_node1_demand.csv', index_col=[0])
        real_leak_demand1.index = pd.to_datetime(real_leak_demand1.index)

        real_leak_dem = real_leak_demand0.values + real_leak_demand1.values
        fig = plt.figure(1)
        ax = fig.add_subplot(2,1,1)
        ax.set_title("Scenario 324: 2 leak nodes, 4 unmeasured nodes,\n 4 fraud nodes, -3% demand measurement error\n(a) Water loss and real leak demand")
        ax.set_ylabel("Water Loss/Leak Demand (L/s)")
        xvals = np.arange(len(loss))*0.5
        ax.plot(xvals, loss, color='g', label='Water Loss')
        ax.plot(xvals,real_leak_dem, color='r', label='Real leak demand')
        ax.legend(loc='upper left')

        ax = fig.add_subplot(2,1,2)
        ax.set_title("(b) Water loss and real leak demand (zoom)")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Water Loss/Leak Demand (L/s)")
        xvals = np.arange(len(loss))*0.5
        ax.plot(xvals, loss, color='g', label='Water Loss')
        ax.plot(xvals,real_leak_dem, color='r', label='Real leak demand')
        ax.axis([5580, 5700, 160, 220])
        ax.legend(loc='upper left')
        plt.show()

        node_press = pd.read_csv(OD_folder_wntr +'/Node_pressures.csv', index_col=[0])
        node_press.index = pd.to_datetime(node_press.index)

        link_flows = pd.read_csv(OD_folder_wntr +'/Link_flows.csv', index_col=[0])
        link_flows.index = pd.to_datetime(link_flows.index)

        fig = plt.figure(2)
        ax = fig.add_subplot(2,1,1)
        ax.set_title("Scenario 324: Difference between real \n and estimated pressures \n (a) Node 145, near leakage")
        ax.set_ylabel("Pressure (m)")
        ax.plot(xvals, node_press.loc[:,'145'], color='blue', label='Node 145 - Real')
        ax.plot(xvals, ODest_node_press.loc[:,'145'], color='red', label='Node 145 - Est', linestyle='dashed')
        ax.axis([5550, 5650, -20, 65])
        ax.legend(loc='lower left')
        
        ax = fig.add_subplot(2,1,2)
        ax.set_title("(b) Node 215, far from leakage")        
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Pressure (m)")
        ax.plot(xvals, node_press.loc[:,'215'], color='blue', label='Node 215 - Real')
        ax.plot(xvals, ODest_node_press.loc[:,'215'], color='red', label='Node 215 - Est', linestyle='dashed')
        ax.axis([5550, 5650, 35, 50])
        ax.legend(loc='lower left')

        fig = plt.figure(3)
        ax = fig.add_subplot(2,1,1)
        ax.set_title("Scenario 324: Nodes with measured pressures \n (a) Node 1 (Tank1)")
        ax.set_ylabel("Pressure (m)")
        ax.plot(xvals, node_press.loc[:,'1'], color='blue', label='Node 1 - Real')
        ax.plot(xvals, ODest_node_press.loc[:,'1']+1-40.20312, color='red', label='Node 1 - Est', linestyle='dashed')
        ax.axis([5580, 5630, 0, 7])
        ax.legend(loc='upper left')

        ax = fig.add_subplot(2,1,2)
        ax.set_title("(b) Node 61 (Pump 335 end node)")
        ax.set_ylabel("Pressure (m)")
        ax.set_xlabel("Time (hours)")
        ax.plot(xvals, node_press.loc[:,'61'], color='blue', label='Node 61 - Real')
        ax.plot(xvals, ODest_node_press.loc[:,'61'], color='red', label='Node 61 - Est', linestyle='dashed')
        ax.axis([5580, 5630, 35, 95])
        ax.legend(loc='upper left')

        fig = plt.figure(4)
        ax = fig.add_subplot(3,1,1)
        ax.set_title("Scenario 324: Difference between real \n and estimated flows \n (a) Pump 335")
        ax.set_ylabel("Flow rate (L/s)")
        ax.plot(xvals, link_flows.loc[:,'335'], color='blue', label='Pump 335 - Real')
        ax.plot(xvals, ODest_link_flows.loc[:,'335'], color='red', label='Pump 335 - Est', linestyle='dashed')
        ax.axis([5550, 5650, 0, 1000])
        ax.legend(loc='upper left')
        
        ax = fig.add_subplot(3,1,2)
        ax.set_title("(b) Pipe 50 - Tank 2 output")        
        ax.set_ylabel("Flow rate (L/s)")
        ax.plot(xvals, link_flows.loc[:,'50'], color='blue', label='Pipe 50 - Real')
        ax.plot(xvals, ODest_link_flows.loc[:,'50'], color='red', label='Pipe 50 - Est', linestyle='dashed')
        ax.axis([5550, 5650, -80, 40])
        ax.legend(loc='lower left')

        ax = fig.add_subplot(3,1,3)
        ax.set_title("(b) Pipe 40 - Tank 1 output")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Flow (L/s)")
        ax.plot(xvals, link_flows.loc[:,'40'], color='blue', label='Pipe 40 - Real')
        ax.plot(xvals, ODest_link_flows.loc[:,'40'], color='red', label='Pipe 40 - Est', linestyle='dashed')
        ax.axis([5550, 5650, -300, 120])
        ax.legend(loc='lower left')

        plt.show()

    if plot_graphs==True:
        fig = plt.figure(1)
        ax = fig.add_subplot(1,1,1)
        ax.set_ylabel("Water Loss(L/s)")
        xvals = np.arange(len(loss))*0.5
        ax.plot(xvals, loss, color='black', label='Measured \n  water Loss')
        ax.legend(loc='upper left')
        
        plt.show()

        detection_labels = pd.read_csv(DATASET_FOLDER_NAME +'/Detection Results/leak_det_labels_sc_'+str(scNum)+'.csv', index_col=[0])
        detection_labels.index = pd.to_datetime(detection_labels.index)

    #Leak Detection and Localization Parameters and variables
    init_big_value = 100000000
    historical_min_loss=init_big_value
    num_complete_days_before_leak = 4
    initial_historical_period = round(num_complete_days_before_leak*24*60/sim_step_minutes)
    historical_max_loss=init_big_value
    overshoot_h = 0.0
    overshoot_l = 0.2
    anomaly_counter = 0
    anomaly_timer=0
    anomaly_counter_threshold = 4
    anomaly_timer_threshold = 16
    stats_sample_period = int(7*60/sim_step_minutes)
    j=-1
    leak_det1=np.zeros(len(timeStamp))
    current_min_loss=0
    df_min_loss = pd.DataFrame(columns = ['Min_loss'])
    df_max_loss = pd.DataFrame(columns = ['Max_loss'])
    num_guess_links = 10
    guess = pd.Series(index=wn.pipe_name_list, dtype=np.float64)
    leak_loc_time_window = 0
    il_delay = 0

    for i in range(0,len(timeStamp),1):
        #Calculates min and max loss in initial historic period and subsequently repeats after each stats_sample_period
        if i<initial_historical_period:
            if i==initial_historical_period-1:
                k = loss.iloc[0:i].idxmin()
                current_min_loss = loss.loc[k]
                df_min_loss.loc[k] = current_min_loss

                k = loss.iloc[0:i].idxmax()
                current_max_loss = loss.loc[k]
                df_max_loss.loc[k] = current_max_loss

                historical_min_loss = df_min_loss.mean(axis=0).iloc[0]
                historical_max_loss = df_max_loss.max(axis=0).iloc[0] 

        else:
            j = (i-initial_historical_period-1)%stats_sample_period
            if j==stats_sample_period-1 and (leak_det1[i-j:i-1].sum())==0:
                k = loss.iloc[i-j-1:i].idxmin()
                current_min_loss = loss.loc[k]
                df_min_loss.loc[k] = current_min_loss

                k = loss.iloc[i-j-1:i].idxmax()
                current_max_loss = loss.loc[k]
                df_max_loss.loc[k] = current_max_loss

        #Get current demand loss
        current_loss = loss.iloc[i]

        #Detects leak
        if i>=initial_historical_period:

            #Plot graph for leak detection understanding
            if plot_graphs==True and detection_labels['Label'].iloc[i]==1:
                fig = plt.figure(10)
                ax = fig.add_subplot(3,1,1)
                ax.set_title('(a) Scenario '+str(scNum)+': Loss demand')
                ax.set_ylabel("Water Loss(L/s)")
                xvals = np.arange(len(loss))*0.5
                ax.plot(xvals, loss, color='blue', label='Measured water Loss')
                ax.plot(xvals, loss*0 + 
                        historical_min_loss + overshoot_l*(historical_max_loss-historical_min_loss),
                        color='red', label='Leakage detection thresholds')
                ax.plot(xvals, loss*0 + 
                        historical_max_loss + overshoot_h*(historical_max_loss-historical_min_loss),
                        color='red')
                ax.plot(xvals, detection_labels['Label']*current_loss*1.2,
                        color='black', label='True label(scaled)', linestyle='dotted')
                ax.plot(xvals, detection_labels['leak_det1']*current_loss*1.2,
                        color='black', label='Detection Result(scaled)', linestyle='dashed')
                ax.legend(loc='upper left')

                ax = fig.add_subplot(3,1,2)
                ax.set_title('(b) Zoom at False Positive detection')
                ax.set_ylabel("Water Loss(L/s) - Zoom")
                xvals = np.arange(len(loss))*0.5
                ax.plot(xvals, loss, color='blue', label='Measured water Loss')
                ax.plot(xvals, loss*0 + 
                        historical_min_loss + overshoot_l*(historical_max_loss-historical_min_loss),
                        color='red', label='Leakage detection thresholds')
                ax.plot(xvals, loss*0 + 
                        historical_max_loss + overshoot_h*(historical_max_loss-historical_min_loss),
                        color='red')
                ax.plot(xvals, detection_labels['Label']*current_loss*1.2,
                        color='black', label='True label(scaled)', linestyle='dotted')
                ax.plot(xvals, detection_labels['leak_det1']*current_loss*1.2,
                        color='black', label='Detection Result(scaled)', linestyle='dashed')

                ax = fig.add_subplot(3,1,3)
                ax.set_title('(c) Zoom at leakage period')
                ax.set_ylabel("Water Loss(L/s) - Zoom")
                ax.set_xlabel("Time (hours)")
                xvals = np.arange(len(loss))*0.5
                ax.plot(xvals, loss, color='blue', label='Measured water Loss')
                ax.plot(xvals, loss*0 + 
                        historical_min_loss + overshoot_l*(historical_max_loss-historical_min_loss),
                        color='red', label='Leakage detection thresholds')
                ax.plot(xvals, loss*0 + 
                        historical_max_loss + overshoot_h*(historical_max_loss-historical_min_loss),
                        color='red')
                ax.plot(xvals, detection_labels['Label']*current_loss*1.2,
                        color='black', label='True label(scaled)', linestyle='dotted')
                ax.plot(xvals, detection_labels['leak_det1']*current_loss*1.2,
                        color='black', label='Detection Result(scaled)', linestyle='dashed')

                plt.show()

            #Leak Detection
            if leak_det1[i-1]==0 and current_loss > historical_max_loss + overshoot_h*(historical_max_loss-historical_min_loss):
                if anomaly_counter==0:
                    il = i
                anomaly_counter+=1

            if anomaly_counter!=0:
                anomaly_timer+=1

            if leak_det1[i-1]==1:
                leak_det1[i]=1
            
            if anomaly_timer>=anomaly_timer_threshold or current_loss < historical_min_loss + overshoot_l*(historical_max_loss-historical_min_loss):
                leak_det1[i]=0
                anomaly_counter=0
                
                historical_min_loss = df_min_loss.min(axis=0).iloc[0]
                historical_max_loss = df_max_loss.max(axis=0).iloc[0] 

                anomaly_timer=0

            if anomaly_counter>=anomaly_counter_threshold :
                leak_det1[i]=1
                anomaly_counter=0

                #Discard min and max values eventually recorded during leak detection time
                dtm_list = df_min_loss.index.to_list()
                for dtm in dtm_list:
                    if dtm>timeStamp[i-anomaly_timer]:
                        df_min_loss.drop(index=dtm)
                
                dtm_list = df_max_loss.index.to_list()
                for dtm in dtm_list:
                    if dtm>timeStamp[i-anomaly_timer]:
                        df_max_loss.drop(index=dtm)

                anomaly_timer=0

                localization_done = False


        #Leak localization
        if leak_det1[i]==1 and localization_done == False:
            localization_done = True
            il = il+il_delay
            il_window = round(leak_loc_time_window*60/sim_step_minutes)+1
               
            wn_test = wn

            for patt_name in wn_test.pattern_name_list:
                patt = wn_test.get_pattern(patt_name)
                if patt_name == '1':
                    patt.multipliers = patt.multipliers[0:il_window]
                else:
                    patt.multipliers = patt.multipliers[il:il+il_window]

            for ctrl_name in wn_test.control_name_list:
                wn_test.remove_control(ctrl_name)

            wn_test.get_link('335').initial_status=wntr.network.base.LinkStatus.Open
            wn_test.get_link('10').initial_status=wntr.network.base.LinkStatus.Open

            # Set time parameters
            ## Energy pattern remove
            wn_test.options.energy.global_pattern = '""'#(matheus)wn.energy.global_pattern = '""'
            # Set time parameters
            wn_test.options.time.duration = (il_window-1)*60*sim_step_minutes
            wn_test.options.time.hydraulic_timestep = 60*sim_step_minutes
            wn_test.options.time.quality_timestep = 0
            wn_test.options.time.report_timestep = 60*sim_step_minutes
            wn_test.options.time.pattern_timestep = 60*sim_step_minutes
            wn_test.options.quality.parameter = "None"
            wn_test.options.hydraulic.demand_model = 'DD'
            wn_test.reset_initial_values()

            a = pd.DataFrame(data = np.ones(len(wn.pipe_name_list))*np.inf, index = wn.pipe_name_list )
            b = pd.DataFrame(data = np.ones(len(wn.pipe_name_list))*np.inf, index = wn.pipe_name_list )
            c = pd.DataFrame(data = np.ones(len(wn.pipe_name_list))*np.inf, index = wn.pipe_name_list )
            d = pd.DataFrame(data = np.ones(len(wn.pipe_name_list))*np.inf, index = wn.pipe_name_list )
            e = pd.DataFrame(data = np.ones(len(wn.pipe_name_list))*np.inf, index = wn.pipe_name_list )

            for pipe_index in range(len(wn.pipe_name_list)):
                pipe_name = wn.pipe_name_list[pipe_index]
                if pipe_name!='10' and pipe_name!='335' and pipe_name!='60':
                    wn_test2 = wntr.morph.split_pipe(wn_test,pipe_name, pipe_name+'test', 
                    'test_node', split_at_point = 0.5)
                    wn_test2.add_pattern('P_test', loss.values[il:il+il_window]-historical_min_loss)
                    wn_test2.get_node('test_node').add_demand(0.001,'P_test')
                    del wn_test2.get_node('test_node').demand_timeseries_list[0]

                    sim = wntr.sim.EpanetSimulator(wn_test2)
                    results = sim.run_sim()
                    
                    # print("Pipe "+pipe_name+" simulation ok")
                    if ((all(results.node['pressure']> 0)) !=True)==True:
                        print("Negative pressures!")

                    flows = results.link['flowrate']
                    flows = flows[:len(timeStamp)]
                    flows = from_si(INP_UNITS, flows, HydParam.Flow)
                    flows = flows.round(results_decimal_digits)
                    flows.index = timeStamp[il:il+il_window]
                    pump10_flows = flows['10']
                    pump335_flows = flows['335']
                    tank1_flows = flows['40']
                    tank2_flows = flows['50']
                    tank3_flows = flows['20']
                                         
                    a.loc[pipe_name]  = ((pump10_flows + ODdem['Lake'].iloc[il:il+il_window])**2).sum()
                    b.loc[pipe_name]  = ((pump335_flows + ODdem['River'].iloc[il:il+il_window])**2).sum()
                    c.loc[pipe_name]  = ((tank1_flows + ODdem['1'].iloc[il:il+il_window])**2).sum()
                    d.loc[pipe_name]  = ((tank2_flows + ODdem['2'].iloc[il:il+il_window])**2).sum()
                    e.loc[pipe_name]  = ((tank3_flows + ODdem['3'].iloc[il:il+il_window])**2).sum()


            guess = a*b*c*d*e

    if isinstance(guess, pd.DataFrame):
        guess = pd.Series(guess.iloc[:,0])

    guess = guess

    guess = guess.sort_values(ascending=True)

    print('Guess pipe list: \n'+str(guess.iloc[0:num_guess_links]))
    selected_pipes = guess.iloc[0:num_guess_links].index

    Labels = pd.read_csv(OD_folder_wntr +'/Labels.csv', index_col=[0])
    Labels.index = pd.to_datetime(Labels.index)

    detection_results_path_os = dataset_folder_os + '\\Detection Results'
    if not os.path.exists(detection_results_path_os):
        os.makedirs(detection_results_path_os)

    flabels = pd.DataFrame(leak_det1)
    flabels['Timestamp'] = timeStamp
    flabels = flabels.set_index(['Timestamp'])
    flabels.columns.values[0]='leak_det1'
    flabels['Label'] = Labels.loc[:,'Label']
    flabels.to_csv(detection_results_path_os+'\\leak_det_labels_sc_'+str(scNum)+'.csv')
    del flabels

    # Calculate accuracy
    accuracy = accuracy_score(Labels, leak_det1)
    print("Accuracy:", accuracy)

    # Calculate precision
    precision = precision_score(Labels, leak_det1)
    print("Precision:", precision)

    # Calculate recall (sensitivity)
    recall = recall_score(Labels, leak_det1)
    print("Recall:", recall)

    # Calculate F1-score
    f1score = f1_score(Labels, leak_det1)
    print("F1-Score:", f1score)

    fresults = dataset_folder_os+'\\Results.csv'

    iter = 0
    is_linked = 0
    is_in = 0
    true_leak_link = str(int(df_results.loc[scNum, 'Leak link 0 Name']))
    for iter in range(num_guess_links):
        if true_leak_link == selected_pipes[iter]:
            is_in=1

        if wn.get_link(true_leak_link).start_node == wn.get_link(selected_pipes[iter]).start_node or wn.get_link(true_leak_link).start_node == wn.get_link(selected_pipes[iter]).end_node or wn.get_link(true_leak_link).end_node == wn.get_link(selected_pipes[iter]).start_node or wn.get_link(true_leak_link).end_node == wn.get_link(selected_pipes[iter]).end_node:
            is_linked=1

        iter+=1

    is_exacly = 0
    if true_leak_link == selected_pipes[0]:
        is_exacly = 1

    df_results.loc[scNum, 'Detector 1 accuracy'] = accuracy
    df_results.loc[scNum, 'Detector 1 precision'] = precision
    df_results.loc[scNum, 'Detector 1 recall'] = recall
    df_results.loc[scNum, 'Detector 1 f1_score'] = f1score
    df_results.loc[scNum, 'Leaky pipe guess list'] = np.array2string(selected_pipes.values)  
    df_results.loc[scNum, 'True localization found?'] = is_exacly
    df_results.loc[scNum, 'True localization is within the list?'] = is_in
    df_results.loc[scNum, 'True localization is linked to pipe within the list?'] = is_linked
            
    return 1
    
if __name__ == '__main__':

    t = time.time()

    NumScenarios = 501
    scArray = range(1, NumScenarios)
    
    for i in scArray:
        if df_results.loc[i, 'Number of Leaks']==1:
            try:
                runScenarios(i)
            except:
                pass

    fresults = dataset_folder_os+'\\Results.csv'
    df_results.to_csv(fresults)

    print('Total Elapsed time is '+str(time.time() - t) + ' seconds.')