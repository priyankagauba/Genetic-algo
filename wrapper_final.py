"""
Main Wrapper class (4 components):
    1. Parameter Initialization
    2. Data Preparation / Data Loading
    3. Clustering
    4. Genetic Algorithm
"""

# =============================================================================
# Importing Libraries
# =============================================================================
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import pickle 
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
pd.options.mode.chained_assignment = None
fullwrapper_start_time = time.time()
# =============================================================================
# Wrapper Parameters
# =============================================================================

""" Component 1: Parameter initilization for modules ( data preparation, clustering, genetic) """

# True if rawdata and agg_data file are not created or segment length changed
data_preparation_flag=False 

# True if filtered data after clustering is present in ram
clustering_flag=True


# =============================================================================
# Data Loading / Preparation Parameters
# =============================================================================
# True if microtrip type based approach to be used else false
flag_microtrip=False
# fixed segment length
trip_minute=5
#Channel txt file location
channel_data_path=r'data\channel_data'         #root_path=r'data\channel_data_gear'      path have to be changed
# path for rawdata and agregated file 
root_path = r'data'

# =============================================================================
# Clustering Parameters
# =============================================================================
#clustering algorithm dictionary
clustering_algo_dict={0:['nmf_clustering',15,0],1:['kmeans_clustering',7,6],2:['dwt_clustering',9,0]}
# index of clustering algorithm dictionary
sel_idx_clus_algo=2

cycle_duration=30

pickel_path=r'data\clustering_data'
# =============================================================================
# Data Preparation  
# =============================================================================
""" Component 2:  Data Preparation / Data Loading
        
        a. Data Preparation:
            
            Parameters:
               1. agg_file_name: aggregated file name
               2. raw_file_name: rawdata file name
               3. channel_data_path: raw text file location
               4. acc_limit, dec_limit: Acceleration, Deceleration limits
               5. speed_limit1, speed_limit2, speed_limit3: Speed Limits
               6. trip_minute : for fixed segment, segment length
               7. flag_microtrip: Flag for fixed segment or microtrip
                                       if true: microtrip 
                                       else:  fixed segment 
                                       
        b. Data Loading:
            
            Parameters:
               1. agg_file_name: aggregated file name
               2. raw_file_name: rawdata file name
"""

if flag_microtrip:
    if 'gear' in channel_data_path:
        rawdata_file_name='complete_data_microtrip_gear.csv'
        aggdata_file_name='aggregated_data_microtrip_gear.csv'
        gear_info='gear'
        
    else:
        rawdata_file_name='complete_data_microtrip.csv'
        aggdata_file_name='aggregated_data_microtrip.csv'
        gear_info=''
    
else:
    if 'gear' in channel_data_path:
        rawdata_file_name='complete_data_'+str(trip_minute)+'min_segment_gear.csv'
        aggdata_file_name='aggregated_data_'+str(trip_minute)+'min_segment_gear.csv'
        gear_info='gear'
    else:
        rawdata_file_name='complete_data_'+str(trip_minute)+'min_segment.csv'
        aggdata_file_name='aggregated_data_'+str(trip_minute)+'min_segment.csv'
        gear_info=''
    
if data_preparation_flag:
    
    from data_preparation_final import data_prep
    
    data_object = data_prep(
                agg_file_name=os.path.join(root_path,aggdata_file_name),
                raw_file_name=os.path.join(root_path,rawdata_file_name),
                channel_data_path=channel_data_path,
                acc_limit=0.1,
                dec_limit=-0.1,
                speed_limit1=20,
                speed_limit2=40,
                speed_limit3=60,
                trip_minute=trip_minute,
                flag_microtrip=flag_microtrip
            )
    rawdata,agg_data=data_object.main()
    agg_data.set_index('mt_id',drop=False,inplace=True)
# =============================================================================
# Data Loading
# =============================================================================
else:   
      
    rawdata = pd.read_csv(os.path.join(root_path,rawdata_file_name))
    
    agg_data = pd.read_csv(os.path.join(root_path,aggdata_file_name))
    agg_data.set_index('mt_id',drop=False,inplace=True)

# =============================================================================
# Data Filtering
# =============================================================================
if flag_microtrip:
    # filtering microtrips in which vehicle speed 0 for 99% duration
    agg_data=agg_data[~((agg_data['idle_percent']>0.99))] 
    rawdata=rawdata[rawdata['mt_id'].isin(agg_data['mt_id'])]
     
else:
    # filtering microtrips in which vehicle speed 0 for 99% duration and microtrips which have duration less than trip minutes
    agg_data=agg_data[~((agg_data['idle_percent']>0.99) | (agg_data['duration']<(trip_minute*60-1)))] 
    rawdata=rawdata[rawdata['mt_id'].isin(agg_data['mt_id'])]
    
    
#print(len(rawdata),rawdata.head(),len(agg_data),agg_data.head())

# =============================================================================
# Clustering
# =============================================================================
"""     
Component 3:  Clustering
            
            Parameters:
                a. rawdata: Rawdata file
                b. agg_data: Aggregated file
                c. clustering_algo, num_clusters, num_components: Clustering algorithm and parameters
                d. num_data_to_select: sampling size 
                e. cycle_duration : Cycle duration to be generated
                
            Output:
                a. pop_raw_df: rawdata with cluster column
                b. pop_agg_df: agg_data with cluster column
                c. target_vector: cost function vector for genetic algorithm
                d. target_matrix: cost function matrix for genetic algorithm
                e. starting_idx: starting cluster id for cycle generation
                f. transition_matrix: transition from one cluster to other 
                g. rawdata_dict: dictionary containing mt_id and respective clusters
                h. clusters_dict: dataframe containing mt_id and respective clusters
                i. dist_df: dataframe containing clusters, mt_id and rank within cluster
                j. filtered_raw_df, filtered_agg_df: filtered rawdata and aggregated data
                k. validity_matrix: matrix with valid speed limits transition from one mt_id to other
                
"""

if clustering_flag:
    print('Clustering started')
    start_time_clust=time.time()
    from clustering_algo import clustering
    
    cluster_object= clustering(
                rawdata=rawdata,
                agg_data=agg_data,
                clustering_algo=clustering_algo_dict[sel_idx_clus_algo][0],           #0: nmf_clustering, 1: kmeans_clustering, 2: dwt_clustering
                num_clusters=clustering_algo_dict[sel_idx_clus_algo][1],
                num_components=clustering_algo_dict[sel_idx_clus_algo][2],
                cycle_duration=cycle_duration,
                gear_info=gear_info
            )
    
    pop_raw_df, pop_agg_df,target_vector, target_matrix, class_summary, starting_idx, transition_matrix, clusters_dict, dist_df=cluster_object.main()
    print(class_summary)
    num_data_to_select=30 if class_summary['count'].min()>30 else class_summary['count'].min()
    print(num_data_to_select)
    #population sampling based on rank within cluster and selecting only top 30 elements from every cluster
    filtered_raw_df,filtered_agg_df=cluster_object.population_selection(pop_raw_df,pop_agg_df,dist_df,num_data_to_select)   
    
    validity_matrix,filtered_raw_df=cluster_object.validity_matrix_func(filtered_raw_df)
    
    clustering_data_dict={'pop_raw_df':pop_raw_df,'pop_agg_df':pop_agg_df,'target_vector':target_vector,
                          'target_matrix':target_matrix,'starting_idx':starting_idx,
                          'transition_matrix':transition_matrix,'clusters_dict':clusters_dict,'dist_df':dist_df,
                          'clustering_algo':clustering_algo_dict[sel_idx_clus_algo][0],
                          'data_segment':rawdata_file_name[14:].split('.')[0],
                          'num_data_to_select':num_data_to_select,
                          'gear_info':gear_info
                          } 
    
    
    
    pickle_final_path=os.path.join(pickel_path,clustering_algo_dict[sel_idx_clus_algo][0]+'_data_dict_'+str(clustering_algo_dict[sel_idx_clus_algo][1])+'_'+rawdata_file_name[14:].split('.')[0]+'.pkl')
    print('Saving data at ',pickle_final_path)
    location=open(pickle_final_path,'wb')
    pickle.dump(clustering_data_dict,location)
    location.close()

    class_summary['clusters']=class_summary.index
    for i in range(len(pop_agg_df.clusters.unique())):
                print('cluster #: ', i)
                if (class_summary[class_summary.clusters==i]['count']>=5).bool():
                    x_tmp=filtered_agg_df[filtered_agg_df['clusters']==i]['mt_id']
                    idx_1,idx_2, idx_3, idx_4, idx_5  = np.random.choice(x_tmp, 5, replace=True)
                    plt.plot(filtered_raw_df[filtered_raw_df['mt_id']==idx_1]['vehicle_speed'].values)
                    plt.plot(filtered_raw_df[filtered_raw_df['mt_id']==idx_2]['vehicle_speed'].values)
                    plt.plot(filtered_raw_df[filtered_raw_df['mt_id']==idx_3]['vehicle_speed'].values)
                    plt.plot(filtered_raw_df[filtered_raw_df['mt_id']==idx_4]['vehicle_speed'].values)        
                    plt.plot(filtered_raw_df[filtered_raw_df['mt_id']==idx_5]['vehicle_speed'].values)    
                    plt.show()

    
    stop_time_clust=time.time()
    print("Time taken in clustering : ",stop_time_clust-start_time_clust)
    

    # =============================================================================
    # Pickel save clustering data
    # =============================================================================
    
    

# =============================================================================
# Full code run time
# =============================================================================
fullwrapper_stop_time = time.time()
fullwrapper_iter_duration_ = fullwrapper_stop_time - fullwrapper_start_time
print("Full code run time  : ",fullwrapper_iter_duration_)


        
