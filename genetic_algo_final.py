import pandas as pd
import numpy as np
import random
import copy
from scipy import spatial
import time
import pickle
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
import scipy.spatial as sp 
import os
import datetime
import re
from stops_km_module import distance_attribute
"""
Solve the 30 min sequence with genetic algorithm

Objective: min distance from one month summary vector and speed-accel histogram
sequence population from the clusters 
---- problem to be solved:
--- find an optimal sequence of microtrips from cluster ...
-- such that their combination have least cost (in objective) 
-- constraint: total duration ~ 30 mins
-- constraint: sequence starts with small microtrip
-- constraint: total number of stops between a range (example: 7 to 20)
"""

#####################################
## raw data with clusters is stored in raw_df
# for a selected cluster first randomly select a microtrip
# compute score from raw data of selected microtrip
# for example  - sample seq: [16, 15, 5, 20, 0, 4, 20]


class genetic_algo_for_clusters(object):
    """vanila implementation of genetic algorithm
    constraints: 
        1. total duration of sequence in given duration range
        2. total number of stops in given stop range
    objective:
        1. Average speed of sequence ~ target average speed
    ... 
    """
    def __init__(self,
                 seq_pop = None, 
                 raw_df = None,#Filtered Rawdata file (only top 30 elements from every cluster)
                 agg_df = None,#Filtered Aggregated file (only top 30 elements from every cluster)
                 target_vector = None,#cost function vector for genetic algorithm
                 target_matrix = None,#cost function matrix for genetic algorithm
                 duration_range = None,# cycle duration range 
                 size_range = None,# cycle size range for fixed segment 
                 start_seq_idx = None,#starting cluster id for cycle generation
                 num_samples = None,#number of samples in population
                 num_iterations = None,# number of generations
                 number_of_crossovers = None,#number of crossover
                 max_iter_crossover = None,# maximum number of iterations for crossover
                 prob = None,# probability for mutation
                 max_iter_mutation = None,# maximum number of iterations for mutation
                 number_of_winners_to_keep = None,# keep members of previous sequence
                 population_size = None,#population of size created
                 fitting_function=None,#cost function
                 rawdata_dict=None,#dictionary containing mt_id and respective clusters
                 validity_matrix=None,#matrix with valid speed limits transition from one mt_id to other
                 transition_matrix=None,#transition from one cluster to other
                 pickle_genetic_dict=None,#pickle file save path
                 gear_info=None
                 ):
        self.seq_pop = seq_pop
        self.raw_df = raw_df
        self.agg_df = agg_df
        self.target_vector = target_vector
        self.target_matrix = target_matrix
        self.duration_range = duration_range
        self.size_range = size_range
        self.start_seq_idx = start_seq_idx
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.number_of_crossovers = number_of_crossovers
        self.max_iter_crossover = max_iter_crossover
        self.prob = prob
        self.max_iter_mutation = max_iter_mutation
        self.number_of_winners_to_keep = number_of_winners_to_keep
        self.population_size = population_size
        self.fitting_function=fitting_function
        self.rawdata_dict=rawdata_dict
        self.validity_matrix=validity_matrix
        self.transition_matrix=transition_matrix
        self.pickle_genetic_dict=pickle_genetic_dict
        self.gear_info=gear_info
        if self.gear_info:
            self.sample_cycle_freq=60
        else:
            self.sample_cycle_freq=600

    def random_map_microtrip_to_cluster(self):
        """
        method to create a mapping b/w cluster and microtrips
        """
        cluster_microtrip_map = {}
        for seq_id in self.seq_pop:
            rnd_mt_id = np.random.choice(self.agg_df[self.agg_df['clusters']==seq_id]['mt_id'])
            
            cluster_microtrip_map[seq_id] = rnd_mt_id
    
        return cluster_microtrip_map

        

    def sample_microtrip_from_cluster_seq(self, sample_seq):
        """
        method to sample microtrip id from given cluster sequence
        agg_df: aggregated data at microtrip level
        """
        microtrip_id_seq = []
        for seq_id in sample_seq:

            # vanila random sample a microtrip id from a cluster id
            ##########################################################
            # add logic to sample smartly
            rnd_mt_id = np.random.choice(self.agg_df[self.agg_df['clusters']==seq_id]['mt_id'])
            
            microtrip_id_seq.append(rnd_mt_id)
    
        return microtrip_id_seq


    def get_raw_data_from_microtrip_seq(self,cluster_seq,microtrip_seq):
        """get a cycle from given microtrip sequence and append to cycle
        """

        if(len(microtrip_seq)==0):
            go=True
            count=0
            while(go):
                microtrip_seq=[]
                count+=1
                sample_list=[]
                
                mt_id=self.rawdata_dict[self.rawdata_dict['clusters']==cluster_seq[0]].index
                valid_mt_id=self.raw_df[self.raw_df['first_speed']<=5]['mt_id'].unique()
                rnd_mt_id=np.random.choice([x for x in mt_id if x in valid_mt_id])
                
                microtrip_seq.append(rnd_mt_id)
                sample_list.append(self.raw_df[self.raw_df['mt_id'].isin([rnd_mt_id])])

                for i in cluster_seq[1:]:
                    
                        valid_mt_id=self.validity_matrix.columns[np.where(self.validity_matrix[self.validity_matrix.index==microtrip_seq[-1]]==1)[1]]
                        cluster_mt_list=self.rawdata_dict[self.rawdata_dict['clusters']==i].index

                        if len([x for x in cluster_mt_list if x in valid_mt_id])>0:
                            rnd_mt_id=np.random.choice([x for x in cluster_mt_list if x in valid_mt_id])
                            
                            microtrip_seq.append(rnd_mt_id)
                            sample_list.append(self.raw_df[self.raw_df['mt_id'].isin([rnd_mt_id])])
                        else:
                            return None,None

                if (len(microtrip_seq)==len(cluster_seq)) | (count>100) :
                    go=False

                    
            if len(microtrip_seq)<len(cluster_seq):
                return None,None
                        
                    
            sample_df=pd.concat(sample_list,ignore_index=True)

        else:
            sample_df = self.raw_df[self.raw_df['mt_id'].isin(microtrip_seq)]
            sample_df = self.reorder_sample_cycle(sample_df, microtrip_seq)
        return sample_df, microtrip_seq

    def reorder_sample_cycle(self, sample_cycle, microtrip_seq):

        sample_df = pd.DataFrame(columns = sample_cycle.columns)
        
        for c_i in microtrip_seq:
            sample_df = sample_df.append(sample_cycle[sample_cycle['mt_id'].isin([c_i])])
        sample_df.reset_index(inplace=True,drop=True)

        return sample_df

    def get_summary_distribution_matrix(self,sample_df):
        """returns the summary matrix of cycle
        """
        bins_speed = [i for i in range(0,120,20)];bins_speed.append(140)
        bins_accel=[i for i in range(-3,4,1)];bins_accel.append(50);bins_accel.insert(0,-50)
        table_accel_cycle=pd.DataFrame()
        table_accel_cycle['veh_binned'] = pd.cut(sample_df['vehicle_speed'], bins_speed)
        table_accel_cycle['accel_binned'] = pd.cut(sample_df['accel'], bins_accel)
        table_accel_cycle['Count']=1
        table_accel_cycle = pd.pivot_table(table_accel_cycle,index=['veh_binned'],values=['Count'],columns=['accel_binned'],aggfunc='sum')
        table_accel_cycle=table_accel_cycle.fillna(0)
        matrix_class_cycle=(table_accel_cycle.values)
        total=np.nansum(matrix_class_cycle)
        matrix_class_cycle=((matrix_class_cycle/total).round(4))*100
        
        binned_dataframe_cycle=pd.DataFrame(0,index=[i for i in range(len(bins_speed)-1)],columns=[i for i in range(len(bins_accel)-1)])
        binned_dataframe_cycle.update(matrix_class_cycle)
        binned_dataframe_cycle=binned_dataframe_cycle.iloc[0:(len(bins_speed)-1),0:(len(bins_accel)-1)]
        return binned_dataframe_cycle.values

    def get_summary_vector(self, sample_df):
        """returns the summary stats of cycle
        """
        cycle_duration,_=self.duration_range
        base_parameters=pd.DataFrame([[sample_df['vehicle_speed'].mean(),sample_df[sample_df['vehicle_speed']>0]['vehicle_speed'].mean(),np.nanpercentile(sample_df['vehicle_speed'],99),np.nanpercentile(sample_df['accel'],99),np.nanpercentile(sample_df['accel'],1),sample_df['flag_idle'].sum(),sample_df['flag_accel'].sum(),sample_df['flag_decel'].sum(),sample_df['flag_cruise'].sum(),sample_df['flag_creeping'].sum()]],
                        columns=['V_Mean','V_Mean_gt0','V_99%','Accel_99%','Accel_1%','%Idle_time','%Acceleration','%Deceleration','%Cruise','%Creeping'])
        reg='^%'
        columns=base_parameters.filter(regex=reg,axis='columns')
        columns=columns.iloc[:,:].div(len(sample_df),axis=0)
        columns=columns.iloc[:,:].mul(100,axis=0)
        base_parameters.update(columns)
        
        list_mt_id=sample_df.mt_id.unique()
        
        dist=0
        for i in list_mt_id:
            dist+=sample_df[sample_df['mt_id']==i]['distance'].agg(lambda x: x.iloc[-1]-x.iloc[0])
            
        base_parameters=pd.concat([base_parameters,pd.DataFrame([[dist]],columns=['Distance_cycleduration'])],axis=1)
        cycle_vector=base_parameters.values.round(2)
        return cycle_vector
      
    def score_cycle(self, sample_vector):
        """returns a scalar score - cosine distance
        """
        score = spatial.distance.cosine(self.target_vector, sample_vector)
        return score


    def score_cycle_matrix(self, sample_matrix, fitting_function):
        """returns a scalar score - row wise cosine distance / chebyshev distance of cycle matrix
        """
        if fitting_function=='row_wise_cosine':
            score=np.nansum(np.diagonal(sp.distance.cdist(self.target_matrix,sample_matrix, 'cosine'))) 
            
        elif fitting_function=='row_wise_chebyshev':
            score=np.nansum(np.diagonal(sp.distance.cdist(self.target_matrix,sample_matrix, 'chebyshev'))) 
            
        elif fitting_function=='custom_chebyshev':
            score=np.nanmax(np.diagonal(sp.distance.cdist(self.target_matrix,sample_matrix, 'chebyshev'))) 
   
        elif fitting_function=='row_wise_canberra':
            row_score=[]
            for i in range(sample_matrix.shape[0]):
                x=spatial.distance.canberra(self.target_matrix[i], sample_matrix[i])
                row_score.append(x)
            score=np.nansum(np.array(row_score))
         
        elif fitting_function=='l2_norm':
            max_value=np.sqrt(np.sum((self.target_matrix*2)**2))
            min_value=0
            differ_matrix = np.absolute(np.subtract(self.target_matrix,sample_matrix))
            score=np.sqrt(np.sum(differ_matrix**2))
            score = (score-min_value)/(max_value-min_value)
            
        elif fitting_function=='braycurtis':   
            score=np.nanmax(np.diagonal(sp.distance.cdist(self.target_matrix,sample_matrix, 'braycurtis'))) 
            
        elif fitting_function=='long_vector':
            score=spatial.distance.cosine(self.target_matrix.ravel(), sample_matrix.ravel())

        return score


    def score_and_validate_cycle(self, sample_cycle):
        """
        compute the objective function for each sample sequence
        1. create cycle from the sample_seq
        2. create summary_vector for sample cycle
        3. compute cosine distance b/w cycle vector and target vector
        output will be a scalar
        """ 
        duration_l, duration_h = self.duration_range
    
        valid_seq = False
        obj_score = np.nan
        matrix_score = np.nan
        
        pickle_path_save=os.path.join(r'output\pickle_files\target_vector',(str(self.pickle_genetic_dict['cycle_duration'])+self.gear_info+'_stops_perkm.pkl'))
        if os.path.isfile(pickle_path_save):
            location=open(pickle_path_save,'rb')
            stops_df=pickle.load(location)
            location.close()
        
        ## cycle duration
        ## assuming raw data is sampled at 0.1 second frequency 
#        print(sample_cycle)
        seq_duration = sample_cycle.shape[0]/self.sample_cycle_freq
        stops_obj=distance_attribute()
        stops_sample_cycle=stops_obj.find_stops_per_km(sample_cycle)
        
        if (seq_duration >= duration_l)&(seq_duration <= duration_h):
            if ((5*int(stops_df['Stops_per_xkm']/5))<=stops_sample_cycle) & ((5*int(stops_df['Stops_per_xkm']/5+1))>=stops_sample_cycle):
                valid_seq = True
        
        ### create summary vector
        sample_vector = self.get_summary_vector(sample_cycle)
        ### create summary matrix
        sample_matrix = self.get_summary_distribution_matrix(sample_cycle)
        
        ## compute cosine distance from target vector
        obj_score = self.score_cycle(sample_vector)
        ## compute absolute sum of difference of base matrix and sample matrix
        matrix_score = self.score_cycle_matrix(sample_matrix,self.fitting_function)

        obj_matrix_score= obj_score + matrix_score
        return obj_matrix_score, valid_seq

        

    def validity_check(self, sample_seq, microtrip_id_seq,sample_cycle):
        """
        checks whether a candidate sequence is valid by comparing..
        ...total duration and sequence length
        ... and starting index
        """

        size_l, size_u = self.size_range
        #duration_l, duration_h = self.duration_range
    
        valid_seq = False
        
        # check the microtrip sequence from validity matrix
        mt_pairs = [microtrip_id_seq[max(i, 0):i + 2] for i in range(len(microtrip_id_seq)-1)]
        valid_count = sum([self.validity_matrix.loc[x, y] for x,y in mt_pairs])
    
        # check validity by sequence size and continuity
        if (len(sample_seq) >= size_l) & (len(sample_seq) <= size_u) & (valid_count==len(microtrip_id_seq)-1):
    
            _, valid_seq = self.score_and_validate_cycle(sample_cycle)
        
        return valid_seq
        
        
    
    def create_sample_sequence(self, num_samples):
        """
        create a random sequence of microtrips from population (seq_pop)...
        such that total duration is in (duration_range) and..
        size of sequence in (size_range)
        """
        
        pop_samples = []
        go = True
    
        while go:
#            print(self.size_range)
            # select the sequence size
            size_l, size_u = self.size_range
            size_k = random.randint(size_l, size_u )
            
            # create the sequence of size_k from populatio seq_pop
            sample_seq = [self.start_seq_idx]
            
            # generated sequence length = 7
            # start with 0th row
            # select next row by multinomial sampling in 0th row
            # next iteration - select next row
# =============================================================================
#             idx_init = self.start_seq_idx
#             for i in range(size_k-1):
#                 rnd_ = np.random.multinomial(1, self.transition_matrix[idx_init])
#                 idx_init = np.where(rnd_>0)[0][0]
#                 sample_seq.append(idx_init)
# =============================================================================
            
            for k in range(size_k-1):
                sample_seq.append(self.seq_pop[random.randint(0,len(self.seq_pop)-1)])
            
            # sample microtrips from passed cluster sequence
#            microtrip_id_seq = self.sample_microtrip_from_cluster_seq(sample_seq)
            
            # create cycle from microtrip sequence
            sample_cycle,microtrip_id_seq = self.get_raw_data_from_microtrip_seq(sample_seq,microtrip_seq=[])
            
            if ((sample_cycle is not None) and (microtrip_id_seq is not None)):
                
    
                # check cycle validity
                cycle_score, validity_check_ = self.score_and_validate_cycle(sample_cycle)
                
                # check sequence validity by total duration
                if validity_check_:
                    pop_samples.append((sample_seq, microtrip_id_seq,sample_cycle))
                    
            ## generate samples up to required number of samples then break loop
            if len(pop_samples) >= num_samples:
                go = False
        
        return pop_samples
    
    
    
    
    def _compute_objective(self, sample_seq, microtrip_id_seq,sample_cycle):
        
        #compute the objective function for each sample sequence
        #sum of cosine distance of target vector and cycle vector
        # and row wise chebyshev distance of target matrix and cycle matrix
        #output will be a scalar
         
        obj_score = None
        if self.validity_check(sample_seq, microtrip_id_seq, sample_cycle):
            obj_score, validity_check_ = self.score_and_validate_cycle(sample_cycle)
        
        return obj_score
        
    
    def select_candidate_sequences(self, pop_samples):
        """
        select randomly chosen good candidate for cross-over 
        -- such that candidates with higher relative scores...
        ... are more likely to be selected
        --- returns the index of selected candidate  in pop_samples   
        
        argumenets
        list of tuples
        """
        pop_score = [self._compute_objective(x,y,z) for (x,y,z) in pop_samples]
        pop_score = np.array(pop_score)
        
        # sort the scores
        temp = pop_score.argsort()
        #print(temp)
        ranks = np.empty_like(temp)
        
        # rank each sequence by the sorted order
        ranks[temp] = np.arange(len(pop_score))
        
        fitness = [len(ranks) - x for x in ranks]
        
        cum_scores = copy.deepcopy(fitness)
        
        for i in range(1,len(cum_scores)):
            cum_scores[i] = fitness[i] + cum_scores[i-1]
        
        probs = [x / cum_scores[-1] for x in cum_scores]
        rand = random.random()
        for i in range(0, len(probs)):
            if rand < probs[i]:
    
                return i
            
    def cross_over(self, a_, b_, max_iter):
        """
        takes two candidates and create new candidates by... 
        mixing them together
        -- new candidate should be a valid candidate
        a, b: tuple (cluster_seq, microtrip_seq)
        crossover on microtrip seq
        """
        # select a random cutpoint of a
        a = copy.deepcopy(a_)
        b = copy.deepcopy(b_)
        go = True
        cnt = 0
        while go:
            cut_a = np.random.choice(range(2, len(a[0])))
            cut_b = np.random.choice(range(2, len(b[0])))
        
            new_a1_cluster = copy.deepcopy(a[0][0:cut_a])
            new_a1_microtrip = copy.deepcopy(a[1][0:cut_a])
            
            new_a2_cluster = copy.deepcopy(b[0][cut_b:])
            new_a2_microtrip = copy.deepcopy(b[1][cut_b:])            
        
            new_b1_cluster = copy.deepcopy(b[0][0:cut_b])
            new_b1_microtrip = copy.deepcopy(b[1][0:cut_b])
            
            new_b2_cluster = copy.deepcopy(a[0][cut_a:])
            new_b2_microtrip = copy.deepcopy(a[1][cut_a:])
            
            cond1=self.validity_matrix.loc[new_a1_microtrip[-1],new_a2_microtrip[0]]
            cond2=self.validity_matrix.loc[new_b1_microtrip[-1],new_b2_microtrip[0]]
            
            cnt += 1
            
            if int(cond1*cond2):
        
                new_a_cluster = np.append(new_a1_cluster, new_a2_cluster)
                new_a_microtrip = np.append(new_a1_microtrip, new_a2_microtrip)
                
                new_b_cluster = np.append(new_b1_cluster, new_b2_cluster)
                new_b_microtrip = np.append(new_b1_microtrip, new_b2_microtrip)
                
                # combine to get original tuple
                sample_cycle_a,_=self.get_raw_data_from_microtrip_seq(new_a_cluster,new_a_microtrip)
                new_a = (new_a_cluster, list(new_a_microtrip),sample_cycle_a)
                sample_cycle_b,_=self.get_raw_data_from_microtrip_seq(new_b_cluster,new_b_microtrip)
                new_b = (new_b_cluster, list(new_b_microtrip),sample_cycle_b)
                new_a_valid = self.validity_check(new_a_cluster, new_a_microtrip,sample_cycle_a)
                new_b_valid = self.validity_check(new_b_cluster, new_b_microtrip,sample_cycle_b)
                
                
            else:
                new_a_valid,new_b_valid=False,False
        
            # break the loop if valid child is found or reached max iter
            if (new_a_valid & new_b_valid) | (cnt >= max_iter):
                go = False
    
        if (new_a_valid & new_b_valid):
            return (new_a, new_b)
        else:
            return (a_, b_)
        
    
    def mutate(self, sample_seq, prob, max_iter):
        """
        reshuffle the sequence randomly
        """
        go = True
#    
#        mutated_seq = copy.deepcopy(sample_seq)
        sample_seq_ = copy.deepcopy(sample_seq)
        cluster_seq,microtrip_seq,sample_cycle=sample_seq_
        initial_cluster_seq=copy.deepcopy(cluster_seq)
        cnt = 0
        while go:
        
            for i in range(1, len(cluster_seq)):
                if random.random() < prob:
                    cluster_seq_temp = self.seq_pop[random.randint(0,len(self.seq_pop)-1)]
                    last_mt_id=microtrip_seq[i-1]
                    valid_mt_id=self.validity_matrix.columns[np.where(self.validity_matrix[self.validity_matrix.index==last_mt_id]==1)[1]]
                    cluster_mt_list=self.rawdata_dict[self.rawdata_dict['clusters']==cluster_seq_temp].index
                    if len([x for x in cluster_mt_list if x in valid_mt_id])>0:
                        cluster_seq[i]=cluster_seq_temp
                        microtrip_seq[i]=np.random.choice([x for x in cluster_mt_list if x in valid_mt_id])
                    else:
                        pass
#                    mutated_seq[1][i] = self.random_cluster_microtrip_map[int(mutated_seq[0][i])]
            cnt += 1
#            print(np.array_equal(cluster_seq,initial_cluster_seq),cnt)
            if np.array_equal(cluster_seq,initial_cluster_seq):
                valid_seq_check=False
                pass
            else:
                sample_cycle,_=self.get_raw_data_from_microtrip_seq(cluster_seq,microtrip_seq)
                valid_seq_check = self.validity_check(cluster_seq, microtrip_seq,sample_cycle)
                new_mutated_seq=(cluster_seq,microtrip_seq,sample_cycle)
#                print('in condition : ',valid_seq_check)
            
            if valid_seq_check | (cnt>=max_iter):
                go = False
        
        if valid_seq_check:
            return new_mutated_seq
        else:
            return sample_seq_
    
        
    def main(self,filtered_raw_df,filtered_agg_df,validity_matrix,num_iter):
        
        last_best_score = 1000000
        start_time = time.time()
        # initialize the population
        
        self.raw_df=filtered_raw_df
        self.agg_df=filtered_agg_df
        self.validity_matrix=validity_matrix
#        print('rawdata: ',self.raw_df.head())
        population=()
        # Previously saved population to be used
        if self.pickle_genetic_dict['initial_population_flag']:
            pickle_path=self.pickle_genetic_dict['pickle_path_initial_pop']
            location=open(pickle_path,'rb')
            population=pickle.load(location)
            print('initial population loaded from previous run ',type(population))
            location.close()
            
        elif num_iter!=0:
            last_population_path=os.path.join(self.pickle_genetic_dict['pickle_path_save_iteration'],self.pickle_genetic_dict['clustering_algo'])
            list_of_files=[i for i in os.listdir(last_population_path) if i.split('_',1)[0].startswith('population')]
            pickle_path = max([os.path.join(last_population_path,i) for i in list_of_files], key=os.path.getctime)
            location=open(pickle_path,'rb')
            population=pickle.load(location)
            print('population loaded from previous iteration: ',pickle_path)
            location.close()
        else:
            print('Generating population')
            population = self.create_sample_sequence(self.num_samples) # create sample population
            print('sample population generated')
        stop_time = time.time()
        iter_duration_ = stop_time - start_time
        
        print("population loading time  : ",iter_duration_)
        performance_check_dict = {}
        
        for itr in range(0, self.num_iterations):
            start_time = time.time()

            new_population = []
            # score each candidate
            scores = [self._compute_objective(x,y,z) for (x,y,z) in population]

            if itr%3==0:
                print('starting iteration ', itr)
                print('score step completed')
            
            # select candidates to create cross-over
            for j in range(0, self.number_of_crossovers):
                sel_1 = self.select_candidate_sequences(population)
                sel_2 = self.select_candidate_sequences(population)    
    
                new_1, new_2 = self.cross_over(population[sel_1], population[sel_2], 
                                               self.max_iter_crossover)
                # check validity of cross-over children
                validity_new_1 = self.validity_check(new_1[0], new_1[1], new_1[2])
                validity_new_2 = self.validity_check(new_2[0], new_2[1], new_2[2])
                if (validity_new_1 & validity_new_2):
#                    print('crossover iteration : ',j)
                
                    new_population = new_population + [new_1, new_2]
            if itr%3==0:
                print('crossover step completed')
            # mutate the sequence
            for j in range(0, len(new_population)):
#                print('mutation iteration : ',j)
                new_m = np.copy(self.mutate(new_population[j], self.prob, 
                              self.max_iter_mutation))
                mutation_valid = self.validity_check(new_m[0], new_m[1], new_m[2])
                if mutation_valid:
                    new_population[j] = new_m  
            if itr%3==0:
                print('mutation step completed')
            # keep members of previous sequence
            new_population += [np.array(population[np.nanargmin(scores)])]
            for j in range(1, self.number_of_winners_to_keep):
                keeper = self.select_candidate_sequences(population)            
                new_population += [np.array(population[keeper])]
            
            # add new random members
            while len(new_population) < self.population_size:
                new_temp = self.create_sample_sequence(1)
                new_population += [np.array(new_temp[0])]
            
            # replace old population with new population
            population = [list(x) for x in new_population] #copy.deepcopy(new_population)

            best_scores = [self._compute_objective(x,y,z) for (x,y,z) in population]
            
    
            # get the best candidate out of the new population
            best_score = np.nanmin(best_scores)
            _best = population[np.nanargmin(best_scores)]
            best_seq = _best[0]
            best_mt_id = _best[1]
            best_cycle=_best[2]
            
            if last_best_score != best_score: 
                performance_check_dict[itr] = (best_score, best_seq, best_mt_id)
                last_best_score=best_score

            stop_time = time.time()
            iter_duration_ = stop_time - start_time
            if itr%3==0:
                print('best_score by min: ',best_score)
                print('time taken in generation %s is %s seconds' % (itr, iter_duration_))
            
         # population pickle file save if flag True after every n iterations
        if self.pickle_genetic_dict['pickle_save_iteration']==True:
            print('storing into pickle ', itr)
            regex_t = re.compile(r'[^0-9]')
            x=re.sub(regex_t, '_', str(datetime.datetime.utcnow())[:19])
            
            summary_dict={'best_seq':best_seq,'best_score': best_score,'best_mt_id': best_mt_id,'perf_dict': performance_check_dict}
            pickle_path=os.path.join(self.pickle_genetic_dict['pickle_path_save_iteration'],self.pickle_genetic_dict['clustering_algo'],'summary_dict_'+self.pickle_genetic_dict['clustering_algo']+'_'+str(num_iter)+'_'+x+'.pkl')
            location=open(pickle_path,'wb')
            pickle.dump(summary_dict,location)
            location.close()
            
            pickle_path2=os.path.join(self.pickle_genetic_dict['pickle_path_save_iteration'],self.pickle_genetic_dict['clustering_algo'],'population_'+self.pickle_genetic_dict['clustering_algo']+'_'+str(num_iter)+'_'+x+'.pkl')
            location2=open(pickle_path2,'wb')
            pickle.dump(population,location2)
            print("population stored :",pickle_path2)
            location2.close()
        
        
        print('Iteration Completed')
        return best_seq, best_score, best_mt_id, performance_check_dict,population,best_cycle
 