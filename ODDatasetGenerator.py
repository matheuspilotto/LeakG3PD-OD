# -*- coding: utf-8 -*-
from __future__ import print_function
import multiprocessing
import wntr
import wntr.network.controls as controls#(matheus)
import numpy as np
import pickle
import os
import shutil
#import csv
import pandas as pd#(matheus)import pandas
import time
import matplotlib.pyplot as plt
from wntr.epanet.util import *
import math
#import sys

benchmark = os.getcwd()[:-17]+'Benchmarks\\'
try:
    os.makedirs(benchmark)
except:
    pass 

# demand-driven (DD) or pressure dependent demand (PDD) 
Mode_Simulation = 'DD'

# Leak types
leak_time_profile = ["big", "small"]
sim_step_minutes = 30

# Set duration in hours
durationHours = 24*365 # One Year
timeStamp = pd.date_range("2024-01-01 00:00", "2024-12-30 23:55", freq=str(sim_step_minutes)+"min")

labelScenarios = []
uncertainty_Topology = 'NO'

#Here user can define the network to simulate as long as the INP file is stored in "\networks" folder
INP = "EPANET Net 3"

print(["Run input file: ", INP])

INP_UNITS = FlowUnits.LPS

Max_unmeasured_junctions = 5
Max_Fraud_junctions = 5
Max_demand_reading_error = 0.05

dataset_folder_os = os.getcwd()+'\\'+INP
dfsummary = pd.DataFrame( {'Number of Leaks': pd.Series(dtype='int'),
                           'Number of Unmeasured Demand Junctions': pd.Series(dtype='int'),
                           'Unmeasured Junctions Namelist': pd.Series(dtype='str'),
                           'Number of Demand Fraud Junctions': pd.Series(dtype='int'),
                           'Fraud Junctions Namelist': pd.Series(dtype='str'),
                           'Demand Reading Error': pd.Series(dtype='float'),
                           'Leak link 0 Name': pd.Series(dtype='str'),
                           'Leak link 0 Max. Demand(LPS)': pd.Series(dtype='float'),
                           'Leak 0 duration(h)': pd.Series(dtype='float'),
                           'Leak link 1 Name': pd.Series(dtype='str'),
                           'Leak link 1 Max. Demand(LPS)': pd.Series(dtype='float'),
                           'Leak 1 duration(h)': pd.Series(dtype='float'),
                           'Cronological relation between Leaks': pd.Series(dtype='int'),
                           'Max. Total Unknown/Fraud demand(LPS)': pd.Series(dtype='float'),
                           'Length uncertainty (%)': pd.Series(dtype='float'),
                           'Diameter uncertainty (%)': pd.Series(dtype='float'),
                           'Roughness uncertainty (%)': pd.Series(dtype='float'),
                           'Max. Equiv. Lengh uncertainty (%)': pd.Series(dtype='float'),
                           'Min. Equiv. Lengh uncertainty (%)': pd.Series(dtype='float')
                           }, index = np.arange(1,501))


# RUN ONLY DEMANDS SCENARIOS
def run_OD_Scenarios(scNum):
    results_decimal_digits = 5 #default value = 5
    #Define paths
    sc_folder_os = dataset_folder_os+'\\Scenario-'+str(scNum)
    leaks_folder_os = sc_folder_os +'\\Leaks'

    #Determine leaky pipes, leak distance to origin nodes and create leak link info archive
        #Define paths
    dataset_folder_wntr = INP
    model_inp_file_wntr = dataset_folder_wntr + '/' + INP + '.inp'
    sc_folder_wntr = dataset_folder_wntr + '/Scenario-'+str(scNum)
    sc_inp_file_wntr = sc_folder_wntr + '/' + INP +'_Scenario-'+str(scNum) + '.inp'
    #     #Get number of leaks in scenario
    wn_scenario = wntr.network.WaterNetworkModel(sc_inp_file_wntr)
    labelScenarios = pd.read_csv(dataset_folder_wntr+'/Labels.csv', header=0, index_col=0)
    nmLeaks = labelScenarios.loc[scNum]['Label']
        #Save leak link information into leak link info archive
    for i in range(0,nmLeaks,1):
        leaknodename = 'leak_node'+str(i)
        leaklinkname = wn_scenario.get_links_for_node(leaknodename, flag='inlet') 
        leaklink = wn_scenario.get_link(leaklinkname[0])  
        fleaklink = open(leaks_folder_os +'\\Leak_link'+str(i)+'_info.csv', 'w')
        fleaklink.write("{} , {}\n".format('Description', 'Value'))
        fleaklink.write("{} , {}\n".format('Leak link name', leaklinkname[0]))
        fleaklink.write("{} , {}\n".format('Start node', leaklink.start_node.name))
        fleaklink.write("{} , {}\n".format('End node', wn_scenario.get_link(leaklinkname[0]+'l').end_node.name))
        fleaklink.write("{} , {}\n".format('Leak distance from start node', leaklink.length))
        fleaklink.close()
        print("OD Scenario "+str(scNum)+" leak link info "+str(i+1)+"/"+str(nmLeaks)+" archive successfully saved")

    #Attributes one year time duration to original model file when scNum=1 
    wn_original = wntr.network.WaterNetworkModel(model_inp_file_wntr)
    if scNum==1:
        ## Energy pattern remove
        wn_original.options.energy.global_pattern = '""'#(matheus)wn.energy.global_pattern = '""'
        # Set time parameters
        wn_original.options.time.duration = durationHours*3600
        wn_original.options.time.hydraulic_timestep = 60*sim_step_minutes
        wn_original.options.time.quality_timestep = 0
        wn_original.options.time.report_timestep = 60*sim_step_minutes
        wn_original.options.time.pattern_timestep = 60*sim_step_minutes
        wn_original.options.quality.parameter = "None"
        wntr.network.io.write_inpfile(wn_original, dataset_folder_os+'\\EPANET Net 3.inp', INP_UNITS, '2.2', False)

    #Select random unmeasured junctions and create unmeasured demand junctions info archive
    Unmeasured_junctions = list()
    funmeasured_junctions = open(sc_folder_os +'\\Unmeasured_demand_junctions_info.csv', 'w')
    funmeasured_junctions.write("{}\n".format('Junction name'))
    num_unmeasured_junctions = int(np.random.uniform(Max_unmeasured_junctions+1))
    for i in range(num_unmeasured_junctions):
        junction_index = int(np.random.uniform(wn_original.num_junctions))
        junction_name = wn_original.junction_name_list[junction_index]
        junction = wn_original.get_node(junction_name)
        while ((junction_name in Unmeasured_junctions) or (junction.base_demand == 0)):
            junction_index = int(np.random.uniform(wn_original.num_junctions))
            junction_name = wn_original.junction_name_list[junction_index]
            junction = wn_original.get_node(junction_name)
        Unmeasured_junctions.append(junction_name)
        funmeasured_junctions.write("{}\n".format(Unmeasured_junctions[i]))
    funmeasured_junctions.close()
    print("OD Scenario "+str(scNum)+" unmeasured demand junctions info archive successfully saved")

    #Select random Fraud junctions and create Fraud junctions info archive
    Fraud_junctions = list()
    fFraud_junctions = open(sc_folder_os +'\\Fraud_junctions_info.csv', 'w')
    fFraud_junctions.write("{}\n".format('Junction name'))
    num_Fraud_junctions = int(np.random.uniform(Max_Fraud_junctions+1))
    for i in range(num_Fraud_junctions):
        junction_index = int(np.random.uniform(wn_original.num_junctions))
        junction_name = wn_original.junction_name_list[junction_index]
        junction = wn_original.get_node(junction_name)
        while ((junction_name in Fraud_junctions) or (junction_name in Unmeasured_junctions) or (junction.base_demand == 0)):
            junction_index = int(np.random.uniform(wn_original.num_junctions))
            junction_name = wn_original.junction_name_list[junction_index]
            junction = wn_original.get_node(junction_name)
        Fraud_junctions.append(junction_name)
        fFraud_junctions.write("{}\n".format(Fraud_junctions[i]))
    fFraud_junctions.close()
    print("OD Scenario "+str(scNum)+" fraud junctions info archive successfully saved")

    #Get scenario original node pressures
    presSc = pd.read_csv(sc_folder_wntr +'/Node_pressures.csv', index_col=[0])
    presSc.index = pd.to_datetime(presSc.index)

    #Remove all pump controls
    for ctrl_name in wn_original.control_name_list:
        wn_original.remove_control(ctrl_name)

    #Change tanks by reservoirs with head patterns to keep head during simulation
    elevation = wn_original.get_node('1').elevation
    tank_head = presSc['1'] + elevation
    wn_original.add_pattern('Tank1',tank_head.to_list())

    pipe_length = wn_original.get_link('40').length
    pipe_diameter = wn_original.get_link('40').diameter
    pipe_roughness = wn_original.get_link('40').roughness
    wn_original.remove_link('40')
    coord = wn_original.get_node('1').coordinates
    wn_original.remove_node('1')
    wn_original.add_reservoir(name = '1', base_head=1, head_pattern = 'Tank1', coordinates=coord)
    wn_original.add_pipe(name = '40', start_node_name='1', end_node_name= '40', length=pipe_length, diameter=pipe_diameter, roughness=pipe_roughness)

    elevation = wn_original.get_node('2').elevation
    tank_head = presSc['2'] + elevation
    wn_original.add_pattern('Tank2',tank_head.to_list())

    pipe_length = wn_original.get_link('50').length
    pipe_diameter = wn_original.get_link('50').diameter
    pipe_roughness = wn_original.get_link('50').roughness
    wn_original.remove_link('50')
    coord = wn_original.get_node('2').coordinates
    wn_original.remove_node('2')
    wn_original.add_reservoir(name = '2', base_head=1, head_pattern = 'Tank2', coordinates=coord)
    wn_original.add_pipe(name = '50', start_node_name='2', end_node_name= '50', length=pipe_length, diameter=pipe_diameter, roughness=pipe_roughness)

    elevation = wn_original.get_node('3').elevation
    tank_head = presSc['3'] + elevation
    wn_original.add_pattern('Tank3',tank_head.to_list())

    pipe_length = wn_original.get_link('20').length
    pipe_diameter = wn_original.get_link('20').diameter
    pipe_roughness = wn_original.get_link('20').roughness
    wn_original.remove_link('20')
    coord = wn_original.get_node('3').coordinates
    wn_original.remove_node('3')
    wn_original.add_reservoir(name = '3', base_head=1, head_pattern = 'Tank3', coordinates=coord)
    wn_original.add_pipe(name = '20', start_node_name='3', end_node_name= '20', length=pipe_length, diameter=pipe_diameter, roughness=pipe_roughness)

    #Remove pumps
    wn_original.remove_link('335')
    wn_original.remove_link('10')
    wn_original.remove_curve('1')
    wn_original.remove_curve('2')

    #Add pipes in place of pumps
    wn_original.add_pipe(name='335',start_node_name = '60', end_node_name = '61', length=0.000001, diameter=10000000000, roughness=0.00000000001, initial_status= 'CLOSED')
    wn_original.add_pipe(name='10',start_node_name = 'Lake', end_node_name = '10', length=0.000001, diameter=10000000000, roughness=0.00000000001, initial_status= 'CLOSED')

    #Pipe 60 properties are changed to make its headloss negligible
    wn_original.get_link('60').length = 0.000001
    wn_original.get_link('60').diameter = 10000000000
    wn_original.get_link('60').roughness = 0.00000000001

    #Add head patterns to reservoirs in order to maintain original measured pressures and controls for pipes to simulate pumps states
    elevation = wn_original.get_node('61').elevation
    river_head_pattern = presSc['61']+elevation
    wn_original.add_pattern('river_head_pattern',river_head_pattern.to_list())
    wn_original.get_node('River').base_head = 1
    wn_original.get_node('River').head_pattern_name = 'river_head_pattern'
    
    press_control_link = wn_original.get_link('335')
    act_open_pressure_control_link = controls.ControlAction(press_control_link, 'status', 1)
    act_close_pressure_control_link = controls.ControlAction(press_control_link, 'status', 0)

    flowSc = pd.read_csv(sc_folder_wntr +'/Link_flows.csv', index_col=[0])
    flowSc.index = pd.to_datetime(flowSc.index)
    pump335flow = flowSc['335']
    pump335status = 0

    pump335status = 0
    for i in range(0,len(pump335flow)):
        if pump335flow.iloc[i]>1 and pump335status==0:
            cond = controls.SimTimeCondition(wn_original, '=', sim_step_minutes*60*i)
            ctrl_name = 'open_pressure_control_link_river'+str(i)
            ctrl = controls.Control(cond, act_open_pressure_control_link, name=ctrl_name)
            wn_original.add_control(ctrl_name, ctrl)
            pump335status = 1

        if pump335flow.iloc[i]<1 and pump335status==1:
            cond = controls.SimTimeCondition(wn_original, '=', sim_step_minutes*60*i)
            ctrl_name = 'close_pressure_control_link_river'+str(i)
            ctrl = controls.Control(cond, act_close_pressure_control_link, name=ctrl_name)
            wn_original.add_control(ctrl_name, ctrl)
            pump335status = 0

    elevation = wn_original.get_node('10').elevation
    lake_head_pattern = presSc['10']+elevation
    wn_original.add_pattern('lake_head_pattern',lake_head_pattern.to_list())
    wn_original.get_node('Lake').base_head = 1
    wn_original.get_node('Lake').head_pattern_name = 'lake_head_pattern'
    
    press_control_link = wn_original.get_link('10')
    act_open_pressure_control_link = controls.ControlAction(press_control_link, 'status', 1)
    act_close_pressure_control_link = controls.ControlAction(press_control_link, 'status', 0)

    for i in range(1,8760,120):
        cond = controls.SimTimeCondition(wn_original, '=', i*3600)
        ctrl_name = 'open_pressure_control_link_lake'+str(i)
        ctrl = controls.Control(cond, act_open_pressure_control_link, name=ctrl_name)
        wn_original.add_control(ctrl_name, ctrl)

        cond = controls.SimTimeCondition(wn_original, '=', (i+2)*3600)
        ctrl_name = 'close_pressure_control_link_lake'+str(i)
        ctrl = controls.Control(cond, act_close_pressure_control_link, name=ctrl_name)
        wn_original.add_control(ctrl_name, ctrl)

    itsok = False
    while itsok != True:
        try:                
            # Path of EPANET Input File
            print("OD Scenario "+str(scNum)+" start")
            
            ###########################################################################  
            ## SET BASE DEMANDS AND PATTERNS from original model and insert reading error
            total_unknown_fraud_demand = np.zeros(len(timeStamp))
            reading_error = round(int(np.random.uniform(-5,6))*0.2*Max_demand_reading_error,2)
            for junction_name in wn_original.junction_name_list:
                pattern_name = 'P_'+junction_name
                base_dem = wn_scenario.get_node(junction_name).base_demand
                orig_pattern = wn_scenario.get_pattern(pattern_name)
                orig_multipliers = orig_pattern.multipliers

                if (junction_name in Unmeasured_junctions):#if unmeasured, base demand equals zero
                    wn_original.get_node(junction_name).add_demand(0, 'None')

                    total_unknown_fraud_demand += base_dem*orig_multipliers
                else:
                    if (junction_name in Fraud_junctions):#if fraud, pattern multipliers are changed within an aleatory period
                        fraud_factor = np.random.uniform(0.5,1)
                        fraud_start_index = int(np.random.uniform(0,orig_multipliers.size))
                        fraud_end_index = int(np.random.uniform(fraud_start_index,orig_multipliers.size))
                        fraud_multipliers = orig_multipliers
                        fraud_multipliers[fraud_start_index:fraud_end_index] = [r*fraud_factor for r in fraud_multipliers[fraud_start_index:fraud_end_index]]
                        wn_original.add_pattern(pattern_name, fraud_multipliers)
                        wn_original.get_node(junction_name).add_demand(base_dem*(1+reading_error),pattern_name)
                        total_unknown_fraud_demand += base_dem*(orig_multipliers-fraud_multipliers)
                    else:#else, base demand and pattern are copied from scenario model, reading error is inserted
                        wn_original.add_pattern(pattern_name, orig_pattern)
                        wn_original.get_node(junction_name).add_demand(base_dem*(1+reading_error),pattern_name)
                del wn_original.get_node(junction_name).demand_timeseries_list[0]
                
            ## SAVE EPANET OnlyDemands FILE 
            # Write inp file
            wntr.network.io.write_inpfile(wn_original, sc_folder_os+'\\EPANET Net 3_OnlyDemands-'+str(scNum)+'.inp', INP_UNITS, '2.2', False)

            ## RUN SIMULATION
            wn_original.options.hydraulic.demand_model = Mode_Simulation #IMPORTANT! In this case Demand Driven was selected!
            sim = wntr.sim.EpanetSimulator(wn_original)
            results = sim.run_sim()
            print("OD Scenario "+str(scNum)+" simulation ok")
            if ((all(results.node['pressure']> 0)) !=True)==True:
                print("not run")
                scNum = scNum + 1
                itsok = False
                print("OD Scenario "+str(scNum)+" error, negative pressures.")
            
            if results:
                demOD = results.node['demand']
                demOD = demOD[:len(timeStamp)]
                demOD = from_si(INP_UNITS, demOD, HydParam.Flow)
                demOD.index = timeStamp

                demSc = pd.read_csv(sc_folder_wntr +'/Node_demands.csv', index_col=[0])
                demSc.index = pd.to_datetime(demSc.index)

                demOD['1'] = demSc['1']
                demOD['2'] = demSc['2']
                demOD['3'] = demSc['3']
                demOD['River'] = demSc['River']
                demOD['Lake'] = demSc['Lake']

                fdemOD = sc_folder_os+'\\OD_Node_demands.csv'
                demOD = demOD.round(results_decimal_digits)
                demOD.to_csv(fdemOD)            
                del fdemOD, demOD, demSc
                os.remove(os.path.join(sc_folder_os,'Node_demands.csv'))

                flows = results.link['flowrate']
                flows = flows[:len(timeStamp)]
                flows = from_si(INP_UNITS, flows, HydParam.Flow)
                flows = flows.round(results_decimal_digits)
                flows.index = timeStamp
                fflows = sc_folder_os +'\\OD_Estimated_Link_flows.csv'
                flows.to_csv(fflows)            
                del fflows, flows

                pres = results.node['pressure']
                pres = pres[:len(timeStamp)]
                pres = from_si(INP_UNITS, pres, HydParam.Pressure)
                pres = pres.round(results_decimal_digits)
                pres.index = timeStamp
                fpres = sc_folder_os +'\\OD_Estimated_Node pressures.csv'
                pres.to_csv(fpres)            
                del fpres, pres

                #Delete unnecessary archives
                os.remove(os.path.join(sc_folder_os,'Link_flows.csv'))
                os.remove(os.path.join(sc_folder_os,'Node_pressures.csv'))

                print("OD Scenario "+str(scNum)+" OnlyDemand(OD) pressure, flow and demand archives successfully saved")

                max_leak0_demand=0
                max_leak1_demand=0
                leak0_duration=0
                leak1_duration=0
                relation_btwn_leaks=''

                if nmLeaks!=0:
                    leaklink0name = wn_scenario.get_links_for_node('leak_node0', flag='inlet')[0]

                    max_leak0_demand = pd.read_csv(sc_folder_wntr +'/Leaks/Leak_leak_node0_demand.csv', index_col=[0]).iloc[:,0].max(axis=0)
                    leak_info = pd.read_csv(sc_folder_wntr +'/Leaks/Leak_leak_node0_info.csv', index_col=[0])
                    leak0_start = pd.to_datetime(leak_info.loc['Leak Start ',' Value'])
                    leak0_end = pd.to_datetime(leak_info.loc['Leak End ',' Value'])
                    leak0_duration = (leak0_end-leak0_start).total_seconds() / 3600.0

                    #Defines leak link names, duration and relation between leaks to sava in summary file
                    if nmLeaks==2:
                        leaklink1name = wn_scenario.get_links_for_node('leak_node1', flag='inlet')[0]

                        max_leak1_demand = pd.read_csv(sc_folder_wntr +'/Leaks/Leak_leak_node1_demand.csv', index_col=[0]).iloc[:,0].max(axis=0)
                        leak_info = pd.read_csv(sc_folder_wntr +'/Leaks/Leak_leak_node1_info.csv', index_col=[0])
                        leak1_start = pd.to_datetime(leak_info.loc['Leak Start ',' Value'])
                        leak1_end = pd.to_datetime(leak_info.loc['Leak End ',' Value'])
                        leak1_duration = (leak1_end-leak1_start).total_seconds() / 3600.0

                        if leak0_start<=leak1_start:
                            if leak0_end<leak1_start:
                                relation_btwn_leaks = 'Leak0 happens first'
                            else:
                                if leak0_end>=leak1_end:
                                    relation_btwn_leaks = 'Leak1 inside Leak0'
                                else:
                                    relation_btwn_leaks = 'Leak1 overlaps Leak0'
                        else:    
                            if leak1_end<leak0_start:
                                relation_btwn_leaks = 'Leak1 happens first'
                            else:
                                if leak1_end>=leak0_end:
                                    relation_btwn_leaks = 'Leak0 inside Leak1'
                                else:
                                    relation_btwn_leaks = 'Leak0 overlaps Leak1'

                    else:
                        leaklink1name = ''
                else:
                    leaklink0name = ''
                    leaklink1name = ''

                #Calculates max_total_unknown_fraud_demand in LPS
                max_total_unknown_fraud_demand = 1000*total_unknown_fraud_demand.max()

                #Get scenario uncertainties info
                infoScenario = pd.read_csv(sc_folder_wntr+'/Scenario-'+str(scNum)+'_info.csv', header=0, index_col=0)
                length_unc = float(infoScenario.loc['Uncertainty_Length_(%) '][' Value'])
                diameter_unc = float(infoScenario.loc['Uncertainty_Diameter_(%) '][' Value'])
                roughness_unc = float(infoScenario.loc['Uncertainty_Roughness_(%) '][' Value'])
                max_eq_len_unc = round(100*(1+length_unc/100) / (1-roughness_unc/100)**1.85 / (1-diameter_unc/100)**4.87,2)
                min_eq_len_unc = round(100*(1-length_unc/100) / (1+roughness_unc/100)**1.85 / (1+diameter_unc/100)**4.87,2)

                #Append scenario summary information
                dfsummary.loc[scNum] = [nmLeaks, num_unmeasured_junctions, ', '.join(Unmeasured_junctions), num_Fraud_junctions, ', '.join(Fraud_junctions), reading_error, leaklink0name, max_leak0_demand, leak0_duration, leaklink1name, max_leak1_demand, leak1_duration, relation_btwn_leaks, max_total_unknown_fraud_demand, length_unc, diameter_unc, roughness_unc, max_eq_len_unc, min_eq_len_unc]
                print("OD Scenario "+str(scNum)+" finished\n")

                itsok = True

            else:
                itsok = False
                print('Results empty')
        except:
            itsok = False
            print("OD Scenario "+str(scNum)+" exception")
            
    return 1    
        
if __name__ == '__main__':
    t = time.time()
    
    NumScenarios = 501
    scArray = range(1, NumScenarios)
    
    for i in scArray:
        run_OD_Scenarios(i)

    fsummary = dataset_folder_os+'\\Summary.csv'
    dfsummary.to_csv(fsummary)

    print('Total Elapsed time is '+str(time.time() - t) + ' seconds.')

