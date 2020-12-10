from genetic_algo_final import genetic_algo_for_clusters
from clustering_algo import clustering

import numpy as np
import matplotlib.pyplot as plt
import pickle 
import os
import sys

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
# =============================================================================
# Load Clustering data
# =============================================================================

pickel_path=r'data\clustering_data'  
gear_flag=False

#clustering algorithm dictionary
clustering_algo_dict={0:['nmf_clustering',15,0],1:['kmeans_clustering',7,6],2:['dwt_clustering',9,0]}
sel_idx=2

rawdata_type='5min_segment'  # 5min_segment  microtrip


list_of_files=[i for i in os.listdir(pickel_path) if (i.startswith(clustering_algo_dict[sel_idx][0])) &  (rawdata_type in i) & (('_'+str(clustering_algo_dict[sel_idx][1])+'_') in i)]

if gear_flag:
    list_of_files = [s for s in list_of_files if "gear" in s]
else:
    list_of_files = [s for s in list_of_files if "gear" not in s]
if len(list_of_files)!=0: 
    pickle_file_path = max([os.path.join(pickel_path,i) for i in list_of_files], key=os.path.getctime) 
    location=open(pickle_file_path,'rb')
    clustering_data_dict=pickle.load(location)
    print('Data loaded : ',pickle_file_path.split('\\')[-1])
    location.close()
else:
    print('First, run clustering file')
    sys.exit(0)


# =============================================================================
# Genetic Algorithm Parameters
# =============================================================================
     
# number of generations and iterations for genetic algorithm
iterations=2
num_generations=5

cycle_duration=30
duration_range=(cycle_duration,cycle_duration+5)            # cycle duration range 

size_range=(6,7)                                            # cycle size range for fixed segment 


num_samples=20                         #number of samples in population 30, 40
number_of_crossovers = 15              #number of crossover   25   35
max_iter_crossover = 100               # maximum number of iterations for crossover   
prob = 0.1                             # probability for mutation
max_iter_mutation = 100                # maximum number of iterations for mutation
number_of_winners_to_keep = 5          # keep members of previous sequence
population_size = 45                   #population of size created  45 55 65

start_seq_idx=clustering_data_dict['starting_idx']          # starting cluster id for cycle generation
fitting_function='row_wise_chebyshev'  #cost function 


pop_raw_df=clustering_data_dict['pop_raw_df']
pop_agg_df=clustering_data_dict['pop_agg_df']

target_vector=clustering_data_dict['target_vector']            #cost function vector for genetic algorithm
target_matrix=clustering_data_dict['target_matrix']            #cost function matrix for genetic algorithm
clusters_dict=clustering_data_dict['clusters_dict']            #dictionary containing mt_id and respective clusters
transition_matrix=clustering_data_dict['transition_matrix']    #transition from one cluster to other
dist_df=clustering_data_dict['dist_df']
gear_info=clustering_data_dict['gear_info']
num_data_to_select=clustering_data_dict['num_data_to_select']

initial_population_flag=True                                  # True to use previously created population  

pickle_path_save_iteration=r'output\pickle_files'

#  pickle path file location of initial population to be passed to genetic algorithm
pickle_path_initial_pop=pickle_path_save_iteration+'\dwt_clustering\population_dwt_clustering_17_2019_12_08_14_10_51.pkl'


pickle_genetic_dict={'pickle_save_iteration':True,                                    #flag for population pickle file to be generated after every iterations
                    'initial_population_flag':initial_population_flag,                #flag for initial population
                    'pickle_path_initial_pop':pickle_path_initial_pop,                #initial population path where file to be stored
                    'pickle_path_save_iteration':pickle_path_save_iteration,
                    'clustering_algo':clustering_data_dict['clustering_algo'],
                    'cycle_duration':cycle_duration}


# =============================================================================
# GENETIC ALGORITHM
# =============================================================================    
"""     
Component 4: GENETIC ALGORITHM
            
            Parameters:
                a. filtered_raw_df: Filtered Rawdata file (only top 30 elements from every cluster)
                b. filtered_agg_df: Filtered Aggregated file (only top 30 elements from every cluster)
                c. validity_matrix : matrix with valid speed limits transition from one mt_id to other
                
            Output:   
                a. best_seq: output cycle cluster id sequence
                b. best_score: output cycle cost function score
                c. best_mt: output cycle mt_id sequence
                d. perf_dict: performance dictionary having convergence score
                e. best_population: output cycle generated from population
"""
    
def population_sampling_after_iteration(population,cluster_object):   
        """random addition of mt_id in generated population
        """
        filtered_raw_df,filtered_agg_df=cluster_object.population_selection(pop_raw_df,pop_agg_df,dist_df,num_data_to_select)
        
        mt_id_pop=[]                      # all mt_id present in population
        for i in range(len(population)):
            mt_id_pop.append(population[i][1])
        mt_id_pop = [x for y in mt_id_pop for x in y]
        mt_id_raw_df=list(set(filtered_raw_df['mt_id']))  # random mt_id 
    
        union_set_mt_id=list(set().union(mt_id_pop,mt_id_raw_df))
        
        filtered_raw_df=pop_raw_df[pop_raw_df['mt_id'].isin(union_set_mt_id)]
        filtered_agg_df=pop_agg_df[pop_agg_df['mt_id'].isin(union_set_mt_id)]
        
        validity_matrix,filtered_raw_df=cluster_object.validity_matrix_func(filtered_raw_df)
        
        return filtered_raw_df,filtered_agg_df,validity_matrix


cluster_object= clustering()

if initial_population_flag==True:

    pickle_path=pickle_path_initial_pop
    location=open(pickle_path,'rb')
    population=pickle.load(location)
    location.close()
    
    
    filtered_raw_df,filtered_agg_df,validity_matrix=population_sampling_after_iteration(population,cluster_object)
    
else:
    
    filtered_raw_df,filtered_agg_df=cluster_object.population_selection(pop_raw_df,pop_agg_df,dist_df,num_data_to_select)
    
    validity_matrix,filtered_raw_df=cluster_object.validity_matrix_func(filtered_raw_df)


gac = genetic_algo_for_clusters(
                        size_range = size_range,
                        seq_pop =filtered_agg_df.clusters.unique(),
                        duration_range =duration_range,  
                        start_seq_idx = start_seq_idx,
                        num_samples = num_samples, 
                        num_iterations = num_generations, 
                        number_of_crossovers =number_of_crossovers, 
                        max_iter_crossover = max_iter_crossover, 
                        prob = prob, 
                        max_iter_mutation = max_iter_mutation,
                        number_of_winners_to_keep = number_of_winners_to_keep, 
                        population_size = population_size,
                        agg_df = filtered_agg_df, 
                        raw_df = filtered_raw_df, 
                        target_vector = target_vector,
                        target_matrix = target_matrix,
                        fitting_function=fitting_function,
                        validity_matrix=validity_matrix,
                        rawdata_dict=clusters_dict,
                        transition_matrix=transition_matrix,
                        pickle_genetic_dict=pickle_genetic_dict,
                        gear_info=gear_info
                        )
        
for i in range(iterations):
    best_seq, best_score, best_mt, perf_dict,best_population, best_cycle=gac.main(filtered_raw_df,filtered_agg_df,validity_matrix,i)
    best_vector = gac.get_summary_vector(best_cycle)
    best_matrix=gac.get_summary_distribution_matrix(best_cycle)
    filtered_raw_df,filtered_agg_df,validity_matrix=population_sampling_after_iteration(best_population,cluster_object)   
        
    # =============================================================================
    # CYCLE GRAPH
    # =============================================================================

    if gear_info:
        plt.figure(figsize=(12, 8))
        xx = np.array(best_cycle['vehicle_speed'])
        xy = np.array(best_cycle['gear'])
        fig, ax = plt.subplots()
        plt.plot(xx, label=(len(xx)/600, best_score))
        plt.legend(loc='best')
        ax.tick_params()
        
        # Get second axis
        ax2 = ax.twinx()
        plt.plot(xy, '--g', label='Gear')
        plt.legend(loc='best')
        ax.tick_params()
    else:
        plt.figure(figsize=(12, 8))
        xx = np.array(best_cycle['vehicle_speed'])
        plt.plot(xx, label=(len(xx)/600, best_score))
        plt.legend(loc='best')
    plt.show()
    
    
    print('summary vector of final cycle')
    print(best_vector)
    print('target summary vector')
    print(target_vector)    
    print('summary matrix of final cycle')
    print(np.sum(best_matrix))
    print('summary matrix of target cycle')
    print(np.sum(target_matrix))
    print('Difference of matrices ')
    print(np.sum(np.absolute(best_matrix - target_matrix)))
    
    
    

        
    



# =============================================================================
# 
# regex_t = re.compile(r'[^0-9]')
# x=re.sub(regex_t, '_', str(datetime.datetime.utcnow())[:19])
# pickle_path_save_final_pop=os.path.join(pickle_path_save_iteration,clustering_algo_dict[sel_idx_clus_algo][0],'final_population_'+clustering_algo_dict[sel_idx_clus_algo][0]+'_'+str(num_iterations)+'_'+x+'.pkl')
# location=open(pickle_path_save_final_pop,'wb')
# pickle.dump(best_population,location)
# location.close()
# =============================================================================


# =============================================================================
# 
#         
#         self.size_range=size_range# cycle size range for fixed segment 
#         self.duration_range =duration_range# cycle duration range 
#         self.start_seq_idx = start_seq_idx#starting cluster id for cycle generation
#         self.num_samples = num_samples#number of samples in population
#         self.num_iterations = num_iterations # number of generations
#         self.number_of_crossovers = number_of_crossovers#number of crossover
#         self.max_iter_crossover = max_iter_crossover# maximum number of iterations for crossover
#         self.prob =prob# probability for mutation
#         self.max_iter_mutation = max_iter_mutation# maximum number of iterations for mutation
#         self.number_of_winners_to_keep =number_of_winners_to_keep# keep members of previous sequence
#         self.population_size = population_size #population of size created
#         self.target_vector = target_vector#cost function vector for genetic algorithm
#         self.target_matrix = target_matrix#cost function matrix for genetic algorithm
#         self.fitting_function=fitting_function#cost function
#         self.clusters_dict=clusters_dict#dictionary containing mt_id and respective clusters
#         self.transition_matrix=transition_matrix #transition from one cluster to other
#         self.pickle_genetic_dict=pickle_genetic_dict#pickle file save path
# =============================================================================
