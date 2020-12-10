import pandas as pd
import numpy as np
import os
from functools import reduce
import copy
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
class data_prep:
    def __init__(self,channel_data_path=None,raw_file_name=None,agg_file_name=None,acc_limit=None,dec_limit=None,speed_limit1=None,speed_limit2=None,speed_limit3=None,trip_minute=None,flag_microtrip=None):
        #path to save the aggregated file and complete data file
        self.agg_file_name=agg_file_name
        self.raw_file_name=raw_file_name
        self.channel_data_path=channel_data_path
        self.acc_limit=acc_limit
        self.dec_limit=dec_limit
        self.speed_limit1=speed_limit1
        self.speed_limit2=speed_limit2
        self.speed_limit3=speed_limit3
        self.trip_minute=trip_minute
        self.flag_microtrip=flag_microtrip
        self.sampling_rate=0

    def read_file(self):
        print("Starting")
        with open(self.raw_file_name, 'w',newline='\n') as raw_file, open(self.agg_file_name, 'w',newline='\n') as agg_file:
            agg_file_list = list()
            raw_file_list = list()
            if 'gear' in self.channel_data_path.lower():
                self.sampling_rate=1
                print("########",self.channel_data_path,"########")
                for dirpath, dirname, filenames in os.walk(self.channel_data_path):
                    pass
                
                for filename in filenames:
                    print("********",filename,"**********")
                    file_path = os.path.join(dirpath, filename)
                    temp_data=pd.read_csv(file_path,sep='\t')
                    temp_data=temp_data.drop(0,0)#remove 1 row containing units
                    temp_data=temp_data.loc[:, ~temp_data.columns.str.contains('^Unnamed')]
                    temp_data['engine_speed']=pd.to_numeric(temp_data[temp_data.columns[2]])
                    temp_data['vehicle_speed']=pd.to_numeric(temp_data[temp_data.columns[1]])
                    temp_data['gear']=pd.to_numeric(temp_data[temp_data.columns[3]])
                    temp_data['time']=pd.to_numeric(temp_data[temp_data.columns[0]])
                    
                    temp_data=temp_data.groupby(['time']).mean()
                    temp_data['filename']=filename
                    temp_data.index = pd.to_timedelta(temp_data.index, unit='s')
                    temp_data=temp_data.resample('1S').ffill()
                
                    temp_data['time']=temp_data.index
                    temp_data=temp_data.reset_index(drop=True)
                    temp_data['time']=temp_data['time'].dt.total_seconds()
                    temp_data['distance']=(temp_data['vehicle_speed'])/(len(temp_data)/3600)
                    raw_one_hour=self.rawdata_file(temp_data)

                    agg_one_hour=self.aggregated_file(raw_one_hour)

                    agg_file_list.append(agg_one_hour)
                    raw_file_list.append(raw_one_hour)
                    
                
            else:
                self.sampling_rate=10
                days=os.listdir(self.channel_data_path)
                for i in days:
                    root_path=os.path.join(self.channel_data_path,i)
                    print("########",root_path,"########")
    
                    for dirpath, _, filenames in os.walk(root_path):
                        column_count=0
                        for file_name in filenames:
                            file_path = os.path.join(dirpath, file_name)
                            if os.path.isfile(file_path) and ((os.path.basename(file_path).lower().startswith('ve'))or (os.path.basename(file_path).lower().startswith('engine_speed'))or (os.path.basename(file_path).lower().startswith('total_distance'))):
                                print("********",file_name,"**********")
                                # reading txt files and removing unnamed column
                                temp_data=pd.read_csv(file_path,sep='\t')
                                temp_data=temp_data.drop(0,0)#remove 1 row containing units
                                temp_data=temp_data.loc[:, ~temp_data.columns.str.contains('^Unnamed')]
                                
                                if(file_name.startswith('ve')):
                                    vspeed_data=pd.DataFrame()
                                    vspeed_data['VehV_v']=pd.to_numeric(temp_data['VehV_v'])
                                    vspeed_data['time']=pd.to_numeric(temp_data['time'])
                                    vspeed_data=vspeed_data.groupby(['time']).mean()
                                    column_count+=1
                
                                elif(file_name.startswith('engine_speed')):
                                    espeed_data=pd.DataFrame()
                                    espeed_data['ENG_ENG_SPEED']=pd.to_numeric(temp_data['ENG_ENG_SPEED'])
                                    espeed_data['time']=pd.to_numeric(temp_data['time'])
                                    espeed_data=espeed_data.groupby(['time']).mean()
                                    column_count+=1
    
                                elif(file_name.startswith('total_distance')):
                                    dist_data=pd.DataFrame()
                                    dist_data['GlbDa_lTotDst']=pd.to_numeric(temp_data['GlbDa_lTotDst'])
                                    dist_data['time']=pd.to_numeric(temp_data['time'])
                                    dist_data=dist_data.groupby(['time']).mean()
                                    column_count+=1
    
                                if(column_count==3):
                                    
                                    df=reduce(lambda x,y: pd.merge(x,y,left_index=True, right_index=True), [vspeed_data, espeed_data, dist_data])
                                    
                                    df = df.rename({'VehV_v':'vehicle_speed','ENG_ENG_SPEED':'engine_speed','ENG_IDLE_SPEED_TARGET':'idle_speed','GlbDa_lTotDst':'distance'}, axis=1)
                                    df['filename']=file_name
                                    df.index = pd.to_timedelta(df.index, unit='s')
                                    df=df.resample('0.1S').ffill()
                                    
                                    df['time']=df.index
                                    df=df.reset_index(drop=True)
                                    df['time']=df['time'].dt.total_seconds()
                                    
                                    raw_one_hour=self.rawdata_file(df)
    
                                    agg_one_hour=self.aggregated_file(raw_one_hour)
    
                                    agg_file_list.append(agg_one_hour)
                                    raw_file_list.append(raw_one_hour)
    
        
            agg_data=pd.concat(agg_file_list,axis=0, ignore_index=True,sort=False)
            agg_data.to_csv(agg_file,index=False)
            rawdata=pd.concat(raw_file_list,axis=0, ignore_index=True,sort=False)
            rawdata.to_csv(raw_file,index=False)
        return rawdata,agg_data
            
    def repetition_delete(self,a):
        last_repeated=1
        curr_count=0
        
        for i in range(1,(len(a))):
            if a[i]==a[i-1]:
                curr_count+=1
            else:
                if curr_count==1:
                    a[i-1]=last_repeated
                else:
                    last_repeated=a[i-1]
                    
                curr_count=1
                
        if curr_count==1:
            a[-1]=last_repeated
            
        return  a

    def rawdata_file(self,df):

        df['date_time']=df['filename'].apply(lambda x: x[14:33])
        df['date']=df['date_time'].apply(lambda x:x[:10])
        df['date']=df['date'].str.replace(r'[^0-9]', '-')
        
        df['time2']=df['date_time'].apply(lambda x:x[11:])
        df['time2']=df['time2'].str.replace(r'[^0-9]', '_')
        
        df['day_time']=df['time2']
        df=df.drop(['filename','time2','date_time'],axis=1)
        # giving microtrips unique id
        if self.flag_microtrip:
            s = df.vehicle_speed.eq(0).cumsum().where(df.vehicle_speed.ne(0))
            df['id'] = df.groupby(s).ngroup()+1
            df['id']=df['id'].replace(to_replace=0, method='ffill')
            
        else:
            df['id']=df.index/(self.trip_minute*60*self.sampling_rate)
            df['id']=df.id.astype('int')
        
        df['mt_id'] = df.date+'_'+df.day_time+'_'+df.id.map(str)
        print(df.head(10))
        
        # creating new column acceleration 
        df['accel']=df['vehicle_speed'].diff()/0.36
        # new feature Accel * vehicle speed
        df['a_v_feature']=df['vehicle_speed']*df['accel']
        
        df['flag_idle']=np.where(df['vehicle_speed']==0,1,0)
        df['flag_accel'] = np.where(df['accel']>self.acc_limit,1,0)
        df['flag_decel'] = np.where(df['accel']<self.dec_limit,1,0)
        df['flag_cruise'] = np.where(((df['accel']>self.dec_limit) & (df['accel']<self.acc_limit) & (df['vehicle_speed']>5)),1,0)
        df['flag_creeping'] = np.where(((df['accel']>self.dec_limit) & (df['accel']<self.acc_limit) & (df['vehicle_speed']>0) & (df['vehicle_speed']<=5)),1,0)

        df['vflag_0_20'] = np.where(((df['vehicle_speed']>=0) & (df['vehicle_speed']<self.speed_limit1)),1,0)
        df['vflag_20_40'] = np.where(((df['vehicle_speed']>=self.speed_limit1) & (df['vehicle_speed']<self.speed_limit2)),1,0)
        df['vflag_40_60'] = np.where(((df['vehicle_speed']>=self.speed_limit2) & (df['vehicle_speed']<self.speed_limit3)),1,0)
        df['vflag_60'] = np.where((df['vehicle_speed']>=self.speed_limit3),1,0) 
        
        df['gear']=df.gear.round()
        df['gear']=self.repetition_delete(df['gear'])

        return df
    def percentile(self,n):
        def percentile_(x):
            return np.percentile(x, n)
        percentile_.__name__ = 'Q_%s' % n
        return percentile_
    
    
    def duration(self,x):
        sum_tmp=x.sum()
        sum_tmp=sum_tmp/self.sampling_rate #sampling rate 
        return sum_tmp
      
    def consecutive_array(self,sr):
        tmp1 = pd.DataFrame(columns=['vehicle_speed'])
        tmp1['vehicle_speed'] = sr
        tmp1['diff'] = tmp1['vehicle_speed'].diff()
        tmp1['flag'] = np.where(tmp1['diff']>0, 1, 0)
        tmp1['counter']=tmp1.flag.diff().ne(0).cumsum()
        tmp2 = tmp1.groupby(['flag', 'counter'], as_index=False)['vehicle_speed'].agg('count')
        tmp3=tmp2.groupby('flag')['vehicle_speed'].agg(['max', 'mean','min'])
#        print(tmp3.shape)
        
        return [tmp3.values.ravel()]


    def aggregated_file(self,df):
        
        key=['min','Q_01','Q_05','Q_25','Q_50','mean','Q_75','Q_90','Q_95','Q_99','max']
        values=['min',self.percentile(1),self.percentile(5),self.percentile(25),self.percentile(50),'mean',self.percentile(75),self.percentile(90),self.percentile(95),self.percentile(99),'max']
        summary_dict=dict(zip(key,values))
        
        summary_dict_veh_speed=copy.deepcopy(summary_dict)
        summary_dict_veh_speed['consecutive_array']=self.consecutive_array
        
        final_df=df.groupby('mt_id').agg({'time':'count','vehicle_speed':summary_dict_veh_speed,'accel':summary_dict, 'a_v_feature':summary_dict,'engine_speed':summary_dict,'distance':lambda x:x.iloc[-1]-x.iloc[0],
                           'flag_accel':self.duration,'flag_decel':self.duration,'flag_cruise':self.duration,'flag_creeping':self.duration,'vflag_0_20':self.duration,'vflag_20_40':self.duration,'vflag_40_60':self.duration,'vflag_60':self.duration,'flag_idle':self.duration})
        
        final_df.columns = ['_'.join(col) for col in final_df.columns.values]
        
        final_df.vehicle_speed_consecutive_array= final_df.vehicle_speed_consecutive_array.apply(lambda x: x[0]) 
#        final_df=final_df.join(pd.DataFrame(final_df.vehicle_speed_consecutive_array.tolist(),columns=['flag_0_max', 'flag_0_mean', 'flag_0_min', 'flag_1_max', 'flag_1_mean', 'flag_1_min']))
#        print(final_df.columns)
        final_df=final_df.rename({'time_count':'duration','distance_<lambda>':'distance'},axis=1)
        final_df['duration']=final_df['duration']/self.sampling_rate #sampling rate 
        
        final_df['time_check']=df.groupby('mt_id')['time'].agg(lambda x: (x.iloc[-1] - x.iloc[0]))

        final_df['idle_percent']=final_df['flag_idle_duration']/final_df['duration']
        
        final_df['mt_id']=final_df.index
        print(final_df.head())

        return final_df


    def main(self):
        rawdata,agg_data=self.read_file()
        
        return rawdata,agg_data