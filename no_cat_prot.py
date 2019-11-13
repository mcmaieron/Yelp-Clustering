# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 22:20:22 2019

@author: mcmai
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 10:36:18 2019

@author: mcmai
"""

import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from kmodes.kprototypes import KPrototypes

####preliminary analisys####

#Loads all business in business.json
business = pd.read_json(r'F:\Mestrado\Computacao\KDD\TrabalhoFinal\dados\business.json', lines=True)
# Get names of indexes for which column city has value different from Toronto
indexNames = business[ business['city'] != "Toronto" ].index
# Delete these row indexes from dataFrame
business.drop(indexNames , inplace=True)
# Reset the index after dropping rows
business.reset_index(drop=True, inplace=True)
mask_restaurants = business['categories'].str.contains('Restaurants')

# create a mask for food
mask_food = business['categories'].str.contains('Food')

# apply both masks
restaurants_and_food = business[mask_restaurants & mask_food]
#Verify null data
missing_data = restaurants_and_food.isnull()
# simply drop whole row with NaN in "attributes" column
No_Na= restaurants_and_food.copy()
No_Na.dropna(subset=["attributes"], axis=0, inplace=True)
# Reset the index after dropping rows
No_Na.reset_index(drop=True, inplace=True)

#takes a dataframe as an input, as well as a list of columns that are dictionaries
#takes each column that is a dictionary, and expands it into a series of dummy columns

def create_attributes(df, dictList):
    
    for dictionaryColumn in dictList:
        
        #the attributes column is a string of dictionaries, so one extra step is taken to convert
        if dictionaryColumn == 'attributes':
            expandedColumns = df[dictionaryColumn].apply(pd.Series)
        else:
            expandedColumns = df[dictionaryColumn].apply(pd.Series)
        
        df = pd.concat([df.drop(dictionaryColumn,axis=1), 
                   expandedColumns]
                  ,axis=1)
        
        #df.fillna(value='{}',inplace=True)
        
    return df
l = ['attributes']
exp_rest = create_attributes(No_Na,l)

## Drop columns that does not fit in the domain or that have the same value for 
## all instances
exp_rest.drop(['HairSpecializesIn', 'AcceptsInsurance','BusinessAcceptsCreditCards','RestaurantsCounterService'], axis=1,inplace=True)

## Standardize values that mean the same thing
exp_rest.WiFi.replace(["u'free'","'free'"],'free',inplace=True)
exp_rest.WiFi.replace(["u'paid'", "'paid'"],'paid',inplace=True)
exp_rest.RestaurantsAttire.replace(["u'dressy'","'dressy'"],'dressy',inplace=True)
exp_rest.RestaurantsAttire.replace("u'formal'",'formal',inplace=True)
exp_rest.Alcohol.replace(["u'full_bar'","'full_bar'"],'full_bar',inplace=True)
exp_rest.Alcohol.replace(["u'beer_and_wine'","'beer_and_wine'"],'beer_and_wine',inplace=True)
exp_rest.NoiseLevel.replace(["u'quiet'","'quiet'"],'quiet',inplace=True)
exp_rest.NoiseLevel.replace(["u'loud'","'loud'"],'loud',inplace=True)
exp_rest.NoiseLevel.replace(["u'very_loud'","'very_loud'"],'very_loud',inplace=True)
exp_rest.Smoking.replace("u'no'",'no',inplace=True)
exp_rest.Smoking.replace("u'outdoor'",'outdoor',inplace=True)
exp_rest.Smoking.replace("u'yes'",'yes',inplace=True)


## Replace Null data for False or Mode data
for j in exp_rest:
    if j in [
            'BikeParking','RestaurantsTakeOut','Caters', 'RestaurantsReservations',
            'GoodForKids','OutdoorSeating','RestaurantsDelivery','HasTV',
            'RestaurantsGoodForGroups','RestaurantsTableService','DogsAllowed',
            'WheelchairAccessible','DriveThru','ByAppointmentOnly','GoodForDancing',
            'HappyHour','CoatCheck'
            ]:
       
       exp_rest[j].replace([np.nan,'None'],'False' , inplace=True)
       
    elif j=='RestaurantsPriceRange2':
        exp_rest[j].replace([np.nan,'None'],'2' , inplace=True)        
        
    elif j=='WiFi':
        exp_rest[j].replace([np.nan,"u'no'","'no'"],'no' , inplace=True)
   
    elif j=='BusinessParking':
        exp_rest[j].replace([np.nan,'None' ,''],"{'garage': False, 'street': True, 'validated': False, 'lot': False, 'valet': False}" , inplace=True)
            
    elif j=='RestaurantsAttire': 
        exp_rest[j].replace([np.nan,"u'casual'","'casual'",'None'],'casual' , inplace=True)
    
    elif j=='Alcohol':  
        exp_rest[j].replace([np.nan,"u'none'",'None',"'none'"],'none' , inplace=True)
    
    elif j=='NoiseLevel':
        exp_rest[j].replace([np.nan,"u'average'","'average'",'None'],'average' , inplace=True)
        
    elif j=='Ambience':
        exp_rest[j].replace([np.nan,'None'],"{'romantic': False, 'intimate': False, 'classy': False, 'hipster': False, 'divey': False, 'touristy': False, 'trendy': False, 'upscale': False, 'casual': True}", inplace=True)
   
    elif j=='GoodForMeal': 
        exp_rest[j].replace([np.nan,'None',''],"{'dessert': False, 'latenight': False, 'lunch': False, 'dinner': False, 'brunch': False, 'breakfast': False}", inplace=True)

    elif j=='Music': 
         exp_rest[j].replace([np.nan,'None',''],"{'dj': False, 'background_music': False, 'no_music': True, 'jukebox': False, 'live': False, 'video': False, 'karaoke': False}" , inplace=True)

    elif j=='Smoking': 
        exp_rest[j].replace([np.nan,'None'],'no' , inplace=True)
        
    elif j=='BestNights':  
         exp_rest[j].replace([np.nan,'None'],"{'monday': False, 'tuesday': False, 'friday': True, 'wednesday': False, 'thursday': True, 'sunday': False, 'saturday': True}", inplace=True)
               
    elif j=='DietaryRestrictions':
        exp_rest[j].replace([np.nan,'None'],"{'dairy-free': False, 'gluten-free': False, 'vegan': False, 'kosher': False, 'halal': False, 'soy-free': False, 'vegetarian': False}" , inplace=True)
        

## End of pre-processing
        
##Start K-prototypes algorithm 
#Choose fields that will be considered and ensure things are dtype="category" (cast)
categorical_field_names = [ 'GoodForKids', 'stars','RestaurantsPriceRange2']
for c in categorical_field_names:
    exp_rest[c] = exp_rest[c].astype('category')
#       get a list of the catgorical indicies    
categoricals_indicies = []
for col in categorical_field_names:
        categoricals_indicies.append(categorical_field_names.index(col))
        
#       add non-categorical fields
#
fields = list(categorical_field_names)

##test changing values of lat long to 
#exp_rest['latitude'] = exp_rest['latitude']
#exp_rest['longitude'] = exp_rest['longitude']
fields.append('latitude')
fields.append('longitude')
#
#       select fields
#
data_cats = exp_rest.loc[:,fields]
#
#       normalize continous fields
#
#       essentially compute the z-score
#
#       note: Could use (x.max() - x.min()) instead of np.std(x)
#
columns_to_normalize     = ['latitude', 'longitude']
exp_rest[columns_to_normalize] = exp_rest[columns_to_normalize].apply(lambda x: (x - x.mean()) / np.std(x))
#
#       kprototypes needs an array
#
data_cats_matrix = data_cats.as_matrix()        


#       model parameters
#
init       = 'Cao'                    # init can be 'Cao', 'Huang' or 'random'
n_clusters = 100                          # how many clusters (hyper parameter)
max_iter   = 100                        # default 100

#       get the model
#
kproto = KPrototypes(n_clusters=n_clusters,init=init,verbose=2)
#
#       fit/predict
#
clusters = kproto.fit_predict(data_cats_matrix,categorical=categoricals_indicies)
#
#       combine dataframe entries with resultant cluster_id
#
proto_cluster_assignments = zip(data_cats_matrix,clusters)        


#       Instantiate dataframe to house new cluster data
#
cluster_df = pd.DataFrame(columns=( 'RestaurantsPriceRange2','GoodForKids', 
                                   'stars','latitude', 'longitude',
                                  'cluster_id'))
#
#       load arrays back into a dataframe
#
for array in proto_cluster_assignments:
        cluster_df = cluster_df.append({'GoodForKids':array[0][2], 'stars':array[0][1],
                                    'RestaurantsPriceRange2':array[0][0],'latitude':array[0][3],'longitude':array[0][4],
                                    'cluster_id':array[1]}, ignore_index=True)


cluster_df.cluster_id.value_counts()
##Ploting location## 
#loads US/Canada shapefile
world = gpd.read_file('F:\Mestrado\Computacao\KDD\TrabalhoFinal\dados\shape_mundo\World_Countries.shp')
toronto_boundaries = gpd.read_file(r'F:\Mestrado\Computacao\KDD\TrabalhoFinal\dados\toronto_boundaries\toronto_boundaries.shp')

#Set CRS as 4326 (WGS84)
crs = {'init': 'epsg:4326'}    

# Create a list with the pair of coordinate (LONG,LAT)    
geometry = [Point(xy) for xy in zip (cluster_df["longitude"],cluster_df["latitude"])]

# Create a datafreme with the column geometry and the crs
geo_df = gpd.GeoDataFrame(cluster_df,crs=crs, geometry=geometry)

#Sets the size of the graph
fig,ax = plt.subplots(figsize = (15,15))

#Plots the world shape
#world.plot(ax=ax, alpha=0.4, color="grey")
toronto_boundaries.plot(ax=ax, alpha=0.4, color="grey")
#Plots the points
geo_df.plot(ax=ax, markersize=20, column='cluster_id', cmap='Set1')
plt.title('Estimated number of clusters: ')
plt.show()

print ('elem clus: ', cluster_df.RestaurantsPriceRange2[cluster_df['cluster_id'] == 1])


