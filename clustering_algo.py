# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:26:06 2019

@author: u22v03
"""
from stops_km_module import distance_attribute
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA,NMF
from sklearn.cluster import KMeans
import pywt
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
import os
from yellowbrick.cluster import KElbowVisualizer
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
pd.options.mode.chained_assignment = None

class clustering:
    def __init__(self,
                 rawdata = None,
                 agg_data = None,
                 clustering_algo=None,
                 num_clusters=None,
                 num_components=None,
                 cycle_duration=None,
                 gear_info=None):
        self.rawdata=rawdata
        self.agg_data=agg_data
        self.clustering_algo=clustering_algo
        self.num_clusters=num_clusters
        self.num_components=num_components
        self.cycle_duration=cycle_duration
        self.gear_info=gear_info

    def get_target_vector(self,rawdata):
    
        ###  Target Vector
        base_parameters=pd.DataFrame([[rawdata['vehicle_speed'].mean(),rawdata[rawdata['vehicle_speed']>0]['vehicle_speed'].mean(),np.nanpercentile(rawdata['vehicle_speed'],99),np.nanpercentile(rawdata['accel'],99),np.nanpercentile(rawdata['accel'],1),rawdata['flag_idle'].sum(),rawdata['flag_accel'].sum(),rawdata['flag_decel'].sum(),rawdata['flag_cruise'].sum(),rawdata['flag_creeping'].sum()]],
                        columns=['V_Mean','V_Mean_gt0','V_99%','Accel_99%','Accel_1%','%Idle_time','%Acceleration','%Deceleration','%Cruise','%Creeping'])
        reg='^%'
        columns=base_parameters.filter(regex=reg,axis='columns')
        columns=columns.iloc[:,:].div(len(rawdata),axis=0)
        columns=columns.iloc[:,:].mul(100,axis=0)
        base_parameters.update(columns)
        
        
        pickle_path_save=os.path.join(r'output\pickle_files\target_vector',(str(self.cycle_duration)+self.gear_info+'_stops_perkm.pkl'))
        if os.path.isfile(pickle_path_save):
            location=open(pickle_path_save,'rb')
            stops_df=pickle.load(location)
            location.close()
  
        else:
            
            stops_obj=distance_attribute(rawdata,self.cycle_duration,self.gear_info)
            stops_obj.calculate_stops_per_km()
            location=open(pickle_path_save,'rb')
            stops_df=pickle.load(location)
            location.close()
            
        base_parameters=pd.concat([base_parameters,stops_df['Distance_cycleduration']],axis=1)
        target_vector=base_parameters.values.round(2)
        return target_vector
        
        
    def get_target_matrix(self,rawdata):
        ###Target Matrix
        bins_speed = [i for i in range(0,120,20)];bins_speed.append(140)
        bins_accel=[i for i in range(-3,4,1)];bins_accel.append(50);bins_accel.insert(0,-50)
    #        print(bins_speed,bins_accel)
        table_accel_cycle=pd.DataFrame()
        table_accel_cycle['veh_binned'] = pd.cut(rawdata['vehicle_speed'], bins_speed)
        table_accel_cycle['accel_binned'] = pd.cut(rawdata['accel'], bins_accel)
        table_accel_cycle['Count']=1
        table_accel_cycle = pd.pivot_table(table_accel_cycle,index=['veh_binned'],values=['Count'],columns=['accel_binned'],aggfunc='sum')
        table_accel_cycle=table_accel_cycle.fillna(0)
        matrix_class_cycle=(table_accel_cycle.values)
        total=np.nansum(matrix_class_cycle)
  
        target_matrix=((matrix_class_cycle/total).round(4))*100
        binned_dataframe_cycle=pd.DataFrame(0,index=[i for i in range(len(bins_speed)-1)],columns=[i for i in range(len(bins_accel)-1)])
        binned_dataframe_cycle.update(target_matrix)
        binned_dataframe_cycle=binned_dataframe_cycle.iloc[0:(len(bins_speed)-1),0:(len(bins_accel)-1)]
        return binned_dataframe_cycle 
    
    def transition_matrix_func(self,agg_data):
        
        A=agg_data['clusters']
        A=A.values
        n_cluster=len(agg_data['clusters'].unique())
        TM = [[0]*n_cluster for _ in range(n_cluster)]
        p = [[0]*n_cluster for _ in range(n_cluster)]
    
        for (i,j) in zip(A,A[1:]):
            TM[i][j] += 1
        #     print(i,j)
        #     break
        index=0
        for i in TM:
            s=sum(i)
            for j in range(n_cluster):
                p[index][j]=(round(TM[index][j]/s,4))*100
            index+=1
            
        col=[str(i) for i in range(n_cluster)]
        TM=pd.DataFrame(TM,columns=col)
        transition_matrix=pd.DataFrame(p,columns=col)
        return (transition_matrix.values)/100
    
    
    
    def validity_matrix_func(self,rawdata):
        rawdata['first_speed']=rawdata.groupby(['mt_id'])['vehicle_speed'].transform('first')
        rawdata['last_speed']=rawdata.groupby(['mt_id'])['vehicle_speed'].transform('last')
        rawdata['first_difference']=rawdata.groupby(['mt_id'])['vehicle_speed'].transform(pd.Series.diff)
        rawdata['first_difference']=rawdata.groupby(['mt_id'])['first_difference'].transform('max')
        
        tmp_df=pd.DataFrame()
        tmp_df=rawdata.groupby('mt_id')[['first_speed','last_speed','first_difference']].max()
        
        
        tmp_df['speed1']=tmp_df['last_speed']+(tmp_df['first_difference'])*1.5
        tmp_df['speed2']=tmp_df['last_speed']-(tmp_df['first_difference'])*1.5
        
        tmp_df['speed_range1']=tmp_df['speed1'].apply(lambda x:((int(x/5)+1)*5))
        tmp_df['speed_range2']=tmp_df['speed2'].apply(lambda x:((int(x/5))*5))
        
        
        i_size=tmp_df.shape[0]
        a=np.ndarray(shape=(i_size,i_size))
        a.fill(99)
        for i in range(i_size):
            for j in range(i_size):
                a[i][j]=int((tmp_df['speed_range1'].iloc[i]>=tmp_df['first_speed'].iloc[j])&(tmp_df['speed_range2'].iloc[i]<=tmp_df['first_speed'].iloc[j]))
        
        validity_matrix=pd.DataFrame(a,index=tmp_df.index,columns=tmp_df.index)
        return validity_matrix,rawdata
    

    def population_selection(self,rawdata,agg_data,df,num_data_to_select):
 
        if self.clustering_algo=='nmf_clustering':
            to_select_mt_id = df.groupby('clusters')['dist'].apply(lambda x:x.sort_values(ascending=False)[:(num_data_to_select*2)]).index
            filtered_agg_df = agg_data[agg_data.mt_id.isin(to_select_mt_id.droplevel(0))]
            filtered_agg_df=filtered_agg_df.groupby('clusters', as_index=False).apply(pd.DataFrame.sample, n=num_data_to_select)
            filtered_agg_df.index=filtered_agg_df.index.droplevel(0)
            filtered_raw_df = rawdata[rawdata['mt_id'].isin(filtered_agg_df['mt_id'].unique())]
        
        elif self.clustering_algo=='kmeans_clustering':
            to_select_mt_id=df.groupby('clusters')['dist'].apply(lambda x:x.sort_values()[:(num_data_to_select*2)]).index
            filter_agg_df = agg_data[agg_data.mt_id.isin(to_select_mt_id.droplevel(0))]
            filtered_agg_df=filter_agg_df.groupby('clusters', as_index=False).apply(pd.DataFrame.sample, n=num_data_to_select)
            filtered_agg_df.index=filtered_agg_df.index.droplevel(0)
            filtered_raw_df = rawdata[rawdata['mt_id'].isin(filtered_agg_df['mt_id'].unique())]
        
        else:
            to_select_mt_id=df.groupby('clusters')['dist'].apply(lambda x:x.sort_values()[:(num_data_to_select*2)]).index
            filtered_agg_df = agg_data[agg_data.mt_id.isin(to_select_mt_id.droplevel(0))]
            filtered_agg_df=filtered_agg_df.groupby('clusters', as_index=False).apply(pd.DataFrame.sample, n=num_data_to_select)
            filtered_agg_df.index=filtered_agg_df.index.droplevel(0)
            filtered_raw_df = rawdata[rawdata['mt_id'].isin(filtered_agg_df['mt_id'].unique())]  
            
        return filtered_raw_df, filtered_agg_df
    
    
    def elbow_method(self,data_vals):
      
        model = KMeans(init="k-means++", random_state=10)
        visualizer = KElbowVisualizer(model, k=(3,20))
        visualizer.fit(data_vals)        # Fit the data to the visualizer
     
        return visualizer.elbow_value_,visualizer
        
# =============================================================================
# DWT CLUSTERING  
# =============================================================================
    
    def Energy(self,coeffs, k):
        return np.sqrt(np.sum(np.array(coeffs[-k]) ** 2)) / len(coeffs[-k])
    
    
    def dwt_clustering(self, rawdata, agg_data):
    
        df = copy.deepcopy(rawdata)
        
        df['counter'] =df.groupby('mt_id').cumcount()
        
        df_x= pd.pivot_table(df, values='vehicle_speed', 
                           index='mt_id', 
                           columns = 'counter', aggfunc='first')
        
        X = np.array(df_x)
        
        mask = np.isnan(X)
        
        idx = np.where(~mask,np.arange(mask.shape[1]),0)
        
        np.maximum.accumulate(idx,axis=1, out=idx)
        
        X[mask] = X[np.nonzero(mask)[0], idx[mask]]
        
        energy_list = []
        for i in range(len(X)):
            ceff = pywt.wavedec(X[i], 'db1')
            energy_list.append([self.Energy(ceff, x) for x in range(len(ceff))])
        
        energy_array = np.array(energy_list)
        
        
        # remove nan rows
        rm_idx = np.isnan(energy_array).any(axis=1)
        energy_array = energy_array[~rm_idx]
        df_x=df_x[~rm_idx]
        #tsne_results = TSNE(n_components=2).fit_transform(energy_array)
        self.num_clusters,elbow_graph=self.elbow_method(energy_array)
        # clustering using kmeans
        kmeans = KMeans(n_clusters=self.num_clusters)
        kmeans.fit(energy_array)
        y_l = kmeans.fit_predict(energy_array)
        
        print(np.unique(y_l, return_counts=True))
        
        df_x['clusters']=y_l  
        agg_data=pd.merge(agg_data,df_x['clusters'],how='inner',left_index=True, right_index=True)   
        clusters_dict=agg_data[['mt_id','clusters']]
        
        
        df=pd.DataFrame(energy_array,index=df_x.index)
        df['clusters']=y_l
        cluster_center=kmeans.cluster_centers_
#        print(cluster_center)
        
        x=list()
        for i in range(self.num_clusters):
            x.append(df[df['clusters']==i].iloc[:,:len(ceff)].apply(lambda x:np.linalg.norm(x-cluster_center[i]),axis=1))
            
        df['dist']=np.nanmax(np.array(pd.DataFrame(x).T), axis=1)
        
        filtered_raw_df = pd.merge(rawdata, agg_data[['clusters']], how='inner', left_on='mt_id',right_index=True)
        
        return filtered_raw_df, agg_data, clusters_dict, df, elbow_graph
             
# =============================================================================
# KMEANS CLUSTERING
# =============================================================================
    
    def kmeans(self,rawdata,agg_data):
        # select columns that are to be used in clustering
        select_cols=['duration','idle_percent','a_v_feature_Q_01','a_v_feature_Q_25', 'a_v_feature_Q_50', 'a_v_feature_mean','a_v_feature_Q_99', 'a_v_feature_min',
                    'accel_min','accel_Q_99','accel_mean','accel_Q_25','accel_Q_50', 'accel_Q_75','accel_Q_95',
                    'vehicle_speed_Q_50', 'vehicle_speed_Q_25','vehicle_speed_Q_75', 'vehicle_speed_mean', 'vehicle_speed_Q_99',
                    'flag_accel_duration', 'flag_decel_duration','flag_cruise_duration','flag_idle_duration',
                    'engine_speed_Q_25', 'engine_speed_Q_50', 'engine_speed_Q_75','engine_speed_mean', 'engine_speed_Q_99', 'engine_speed_min']
        
        
        #agg_data = agg_data.dropna(how='any')
        agg_data = agg_data.fillna(0)
        # scale the variables
        min_max_scaler = MinMaxScaler()
        norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(agg_data[select_cols]), 
                                    columns=select_cols, 
                                    index=agg_data.index)
        
        norm_train_array = norm_train_df.values
        
        pca = PCA(n_components=self.num_components)
        principalComponents = pca.fit_transform(norm_train_array)
        
        print('Explained variance : ',pca.explained_variance_ratio_.cumsum())
        
        self.num_clusters,elbow_graph=self.elbow_method(principalComponents)
        
        kmeans = KMeans(n_clusters=self.num_clusters)
        kmeans.fit(principalComponents)
        y_l = kmeans.fit_predict(principalComponents)
        
        #class column in original data
        agg_data['clusters']=y_l
        
        clusters_dict=agg_data[['mt_id','clusters']]
        
        
        df=pd.DataFrame(principalComponents,index=norm_train_df.index)
        df['clusters']=y_l
        cluster_center=kmeans.cluster_centers_
#        print(cluster_center)
        
        x=list()
        for i in range(self.num_clusters):
            x.append(df[df['clusters']==i].iloc[:,:self.num_components].apply(lambda x:np.linalg.norm(x-cluster_center[i]),axis=1))
            
        print(df.head(),len(df),sum([len(x[i]) for i in range(len(x))]))  
        df['dist']=np.nanmax(np.array(pd.DataFrame(x).T), axis=1)
        
        print(rawdata.head(),agg_data.head())
        filtered_raw_df = pd.merge(rawdata, agg_data[['clusters']], how='inner', left_on='mt_id',right_index=True)
        
        return filtered_raw_df, agg_data, clusters_dict, df, elbow_graph

# =============================================================================
# NMF CLUSTERING
# =============================================================================

    def nmf_cluster(self,x_array = None, wide_bin_data= None, agg_df=None,
                    clip_percentile=99,
                    num_clusters = 10, nmf_l1_ratio = 0.5):
        # distribution is very skewed ...
        # ceil all values above 99th percentil
        xx = copy.deepcopy(x_array)
        
        xx = TfidfTransformer().fit_transform(xx)

        #### NMF based clustering - non negative matrix factorization
        nmf = NMF(n_components=num_clusters, random_state=1,
                  alpha=.2, l1_ratio=nmf_l1_ratio).fit_transform(xx)
    
        nmf_clusters = nmf.argmax(axis=1)
        #### highest association clusters
        # nmf_clusters_idx = nmf.argmax(axis=0)
        
        tmp1 = pd.DataFrame()
        tmp1['mt_id'] = wide_bin_data.index
        tmp1['clusters'] = nmf_clusters
        
        tmp2 = pd.merge(agg_df, tmp1, how='inner',left_index=True, right_on='mt_id')
        
        tmp_n= tmp2.groupby('clusters').agg({'mt_id':'count',
                          'duration':'median',
                          'vehicle_speed_mean':'median',
                          'accel_mean':'median'})
        
        return nmf, tmp1, tmp_n
    
    def nmf_clustering(self,rawdata,df):   
        ##### create bin features
#        df=agg_data
        bins_speed = [i for i in range(0,150,5)]
        # print(bins_speed)
        bins_accel=[i for i in range(-10,11,1)]
        bins_accel.append(115);bins_accel.insert(0,-360)
        
        table_accel=pd.DataFrame()
        table_accel['veh_binned'] = pd.cut(rawdata['vehicle_speed'], bins_speed)
        table_accel['accel_binned'] = pd.cut(rawdata['accel'], bins_accel)
        table_accel['Count']=1
        table_accel['mt_id'] = rawdata['mt_id']
        wide_bins = pd.pivot_table(table_accel,index=['mt_id'],
                                   values=['Count'],
                                   columns=['veh_binned', 'accel_binned'],
                                   aggfunc='sum')
        
        # fill na with 0
        wide_bins.fillna(0, inplace=True)
        
        # scale all frequency counts by duration
        # merge with aggregate data to get duration
        wide_bins.columns = range(wide_bins.shape[1])
        wide_bins_duration = pd.merge(wide_bins, df[['duration']], how='inner',
                                      left_index=True, right_index=True)
        
        wide_bins_duration = wide_bins_duration.iloc[:, :-2].div(wide_bins_duration['duration'], axis=0)
        xx = np.array(wide_bins_duration)
        
        nmf, tmp1, tmp_n = self.nmf_cluster(xx, wide_bins_duration, df,
                                 num_clusters=self.num_clusters, nmf_l1_ratio=0.9)
        
        
        # drop the clusters with low duration microtrips - less than 10 seconds
        to_drop_clusters = tmp_n[(tmp_n['duration']<10)&(tmp_n['vehicle_speed_mean']<5)].index
        to_drop_idx = tmp1[tmp1['clusters'].isin(to_drop_clusters)]['mt_id']
        wide_bins_filtered = wide_bins_duration.drop(to_drop_idx.values)
        
        
        
        xx_d = np.array(wide_bins_filtered)
        
        nmf, tmp_cm, tmp_n = self.nmf_cluster(xx_d, wide_bins_filtered, df,
                                 num_clusters=self.num_clusters, nmf_l1_ratio=0.8)
        
        filtered_agg_data = pd.merge(df.drop('mt_id',axis=1), tmp_cm, how='inner',left_index=True, right_on='mt_id')
        
        clusters_dict=filtered_agg_data[['mt_id','clusters']]
        
        tmp_cm['dist']=pd.Series(np.apply_along_axis(np.nanmax, 1, nmf))
        tmp_cm.set_index('mt_id',drop=True,inplace=True)

        filtered_raw_df = pd.merge(rawdata, filtered_agg_data[['clusters','mt_id']], how='inner',on='mt_id')
        filtered_agg_data.index=filtered_agg_data['mt_id']
        return filtered_raw_df, filtered_agg_data, clusters_dict, tmp_cm

# =============================================================================
# CLUSTERING OUTPUT
# =============================================================================
    def class_summary_func(self,agg_data):
        class_summary=agg_data.groupby(by='clusters').agg({'mt_id':'count',
                                                        'duration':'median',
                                                        'vehicle_speed_mean':['mean','std'],
                                                        'vehicle_speed_Q_05':'median',
                                                        'accel_min':'min',
                                                        'accel_Q_05':'median',
                                                        'accel_Q_50':'median',
                                                        'accel_Q_95':'median',
                                                        'accel_max':'max',
                                                        'engine_speed_mean':'mean',
                                                        'engine_speed_Q_50':'median',
                                                        'flag_accel_duration':'median',
                                                        'flag_decel_duration':'median',
                                                        'flag_cruise_duration':'median',
                                                        'vflag_0_20_duration':'median',
                                                        'vflag_20_40_duration':'median',
                                                        'vflag_40_60_duration':'median',
                                                        'vflag_60_duration':'median',
                                                        'flag_idle_duration':'median'
                                                        })

        class_summary.columns=['_'.join(col) for col in class_summary.columns.values]
        class_summary=class_summary.rename({'vehicle_speed_mean_mean':'vspeed_mean','vehicle_speed_mean_std':'vspeed_std','mt_id_count':'count',
            'duration_median':'duration','idle_time_duration_median':'idle_time','accel_Q_05_median':'accel_Q_05','accel_Q_95_median':'accel_Q_95'},axis=1)
        class_summary=class_summary[['count','vspeed_mean','vspeed_std','duration','accel_Q_05','accel_Q_95']]
    
        return class_summary
    
    def main(self):    
            
        target_vector=self.get_target_vector(self.rawdata)
        
        target_matrix=self.get_target_matrix(self.rawdata)
        
        if self.clustering_algo=='nmf_clustering':
            filtered_raw_df, filtered_agg_df, clusters_dict, dist_df=self.nmf_clustering(self.rawdata,self.agg_data)
        
        elif self.clustering_algo=='kmeans_clustering':
            filtered_raw_df, filtered_agg_df, clusters_dict, dist_df, elbow_graph=self.kmeans(self.rawdata, self.agg_data)
            elbow_graph.show()
        
        else:
            filtered_raw_df, filtered_agg_df, clusters_dict, dist_df, elbow_graph=self.dwt_clustering(self.rawdata, self.agg_data)
            elbow_graph.show()
        
        clusters_dict.set_index(['mt_id'],drop=True,inplace=True)
        class_summary=self.class_summary_func(filtered_agg_df)
        
        starting_idx=class_summary['vspeed_mean'].idxmin()
        
        transition_matrix=self.transition_matrix_func(filtered_agg_df)

        return filtered_raw_df, filtered_agg_df,target_vector, target_matrix, class_summary, starting_idx, transition_matrix, clusters_dict, dist_df
        
        
        
