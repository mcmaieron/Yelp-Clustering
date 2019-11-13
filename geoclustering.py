import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
import numpy as np
from kmodes.kprototypes import KPrototypes

####preliminary analisys####

#Loads all business in business.json
business = pd.read_json(r'F:\Mestrado\Computacao\KDD\TrabalhoFinal\dados\business.json', lines=True)

##Looking at attributes##

#Print 5 first rows
business.head(5)

#View all columns names and types
business.dtypes

#Create and save locally a statistic summary of data
summary = business.describe(include="all")
summary
summary.to_csv('F:\Mestrado\Computacao\KDD\TrabalhoFinal\data\summary_rest.csv')

#View all occurrencies in 'attributes' column and its frequencies
attributes = business["attributes"].value_counts()
attributes

#View all occurrencies in 'categories' column and its frequencies
categories = business["categories"].value_counts()
categories

##Filtering only Toronto's business##

# Get names of indexes for which column city has value different from Toronto
indexNames = business[ business['city'] != "Toronto" ].index
# Delete these row indexes from dataFrame
business.drop(indexNames , inplace=True)
# Reset the index after dropping rows
business.reset_index(drop=True, inplace=True)
#Save dataframe as a json file localy
business.to_json(r'F:\Mestrado\Computacao\KDD\TrabalhoFinal\dados\toronto2.json')

#Loads restaurants from Toronto in toronto.json
toronto = pd.read_json(r'F:\Mestrado\Computacao\KDD\TrabalhoFinal\dados\toronto2.json', lines=True)

##pre-processing##

# create a mask for restaurants
mask_restaurants = business['categories'].str.contains('Restaurants')

# create a mask for food
mask_food = business['categories'].str.contains('Food')

# apply both masks
restaurants_and_food = business[mask_restaurants & mask_food]
summary2 = restaurants_and_food.describe(include="all")
summary2.to_csv(r'F:\Mestrado\Computacao\KDD\TrabalhoFinal\dados\summary_rest.csv')
# List all categories
restaurants_and_food['categories'].value_counts()

# Creates a set with all different categories. It is used to find categories 
# thas does not fit in the domain and may interfer in the clustering step
results = set()
restaurants_and_food['categories'].str.lower().str.split().apply(results.update)
results


categoriesToRemove = [
                        'Grocery','Venues & Event Spaces','Drugstores',
                        'Convenience Stores','Beauty & Spas',
                        'Photography Stores & Services','Cosmetics & Beauty Supply',
                        'Discount Store','Fashion','Department Stores',
                        'Gas Stations','Automotive','Music & Video',
                        'Event Planning & Services','Mobile Phones',
                        'Health & Medical','Weight Loss Centers',
                        'Home & Garden','Kitchen & Bath','Jewelry',
                        "Children's Clothing",'Accessories','Home Decor',
                        'Bus Tours','Auto Glass Services','Auto Detailing',
                        'Oil Change Stations', 'Auto Repair','Body Shops',
                        'Car Window Tinting','Car Wash','Gluten-Free',
                        'Fitness & Instruction','Nurseries & Gardening',
                        'Wedding Planning','Embroidery & Crochet',
                        'Dance Schools','Performing Arts','Wholesale Stores'
                        'Tobacco Shops','Nutritionists','Hobby Shops',
                        'Pet Services','Electronics','Plumbing','Gyms',
                        'Yoga','Walking Tours','Toy Stores','Pet Stores',
                        'Pet Groomers','Vape Shops','Head Shops',
                        'Souvenir Shops','Pharmacy','Appliances & Repair'
                        'Wholesalers','Party Equipment Rentals','Tattoo',
                        'Funeral Services & Cemeteries','Sporting Goods',
                        'Dog Walkers','Pet Boarding/Pet Sitting',
                        'Scavenger Hunts','Contractors','Trainers',
                        'Customized Merchandise', 'Dry Cleaning & Laundry',
                        'Art Galleries','Tax Law', 'Bankruptcy Law',
                        'Tax Services', 'Estate Planning Law',
                        'Business Consulting', 'Lawyers', 'Pet Adoption',
                        'Escape Games', 'Animal Shelters',
                        'Commercial Real Estate', 'Real Estate Agents',
                        'Real Estate Services', 'Home Inspectors','Active',
                        'Activities','Antiques','Appliances','Arcades',
                        'Art','Arts','Aooks','Aookstores','Building',
                        'Cards','Care','Casinos','Caterers','Centers',
                        'Churches','Classes','Computer','Education',
                        'Entertainment','Government','Instruments','Massage',
                        'Organizations','Schools','Stationery','Supplements',
                        'Supplies'
                        ]
# Remove the categories
pat = r'\b(?:{})\b'.format('|'.join(categoriesToRemove))
restaurants_and_food['categories'] = restaurants_and_food['categories'].str.replace(pat, '')
restaurants_and_food['categories'] = restaurants_and_food['categories'].str.replace(', ,', ',')

#Verify null data
missing_data = restaurants_and_food.isnull()
missing_data.head(20)

#Count missing values in each column
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")
# simply drop whole row with NaN in "attributes" column
No_Na= restaurants_and_food.copy()
No_Na.dropna(subset=["attributes"], axis=0, inplace=True)
# Reset the index after dropping rows
No_Na.reset_index(drop=True, inplace=True)

missing_data2 = No_Na.isnull()
missing_data2.head(20)
for column in missing_data2.columns.values.tolist():
    print(column)
    print (missing_data2[column].value_counts())
    print("")

restaurants_and_food.to_json(r'F:\Mestrado\Computacao\KDD\TrabalhoFinal\dados\toronto2.json')
# Complete all non liste attributes with 'False'  
attributes = restaurants_and_food['attributes'] 
attributes
#create_attributes

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
exp_rest.columns
exp_rest.head(5)
exp_rest.RestaurantsPriceRange2.mode()


for column in exp_rest.columns[14:42].values.tolist():
    print(column)
    print (exp_rest[column].value_counts())
    print("")
exp_rest.DietaryRestrictions.value_counts()
exp_rest.drop(['HairSpecializesIn', 'AcceptsInsurance','BusinessAcceptsCreditCards','RestaurantsCounterService'], axis=1,inplace=True)
#exp_rest.BikeParking.replace('None','False',inplace=True)
#exp_rest.RestaurantsPriceRange2.replace('None','2',inplace=True)
#exp_rest.WiFi.replace("u'no'" and "'no'",'no',inplace=True)
exp_rest.WiFi.replace(["u'free'","'free'"],'free',inplace=True)
exp_rest.WiFi.replace(["u'paid'", "'paid'"],'paid',inplace=True)
#exp_rest.RestaurantsAttire.replace("u'casual'" and "'casual'" and 'None','casual',inplace=True)
exp_rest.RestaurantsAttire.replace(["u'dressy'","'dressy'"],'dressy',inplace=True)
exp_rest.RestaurantsAttire.replace("u'formal'",'formal',inplace=True)
exp_rest.Alcohol.replace(["u'full_bar'","'full_bar'"],'full_bar',inplace=True)
exp_rest.Alcohol.replace(["u'beer_and_wine'","'beer_and_wine'"],'beer_and_wine',inplace=True)
#exp_rest.Alcohol.replace("u'none'",'none',inplace=True)
#exp_rest.Alcohol.replace('None' and "'none'",'none',inplace=True)
#exp_rest.NoiseLevel.replace("u'average'" and "'average'" and 'None','average',inplace=True)
exp_rest.NoiseLevel.replace(["u'quiet'","'quiet'"],'quiet',inplace=True)
exp_rest.NoiseLevel.replace(["u'loud'","'loud'"],'loud',inplace=True)
exp_rest.NoiseLevel.replace(["u'very_loud'","'very_loud'"],'very_loud',inplace=True)
exp_rest.Smoking.replace("u'no'",'no',inplace=True)
exp_rest.Smoking.replace("u'outdoor'",'outdoor',inplace=True)
exp_rest.Smoking.replace("u'yes'",'yes',inplace=True)

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
        

print ("Shape of the DataFrame: ", exp_rest.shape)
exp_rest.head(3)

#Ensure things are dtype="category" (cast)
categorical_field_names = ['categories', 'GoodForKids', 'stars','RestaurantsPriceRange2']
for c in categorical_field_names:
    exp_rest[c] = exp_rest[c].astype('category')
#       get a list of the catgorical indicies    
categoricals_indicies = []
for col in categorical_field_names:
        categoricals_indicies.append(categorical_field_names.index(col))

data_new=pd.get_dummies(exp_rest, columns=['categories', 'GoodForKids'],drop_first=True)
data_new['categories_Venues & Event Spaces, Food, Event Planning & Services, Restaurants, Breakfast & Brunch, Cafes, Coffee & Tea'].value_counts
#       add non-categorical fields
#
fields = list(categorical_field_names)
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
init       = 'Huang'                    # init can be 'Cao', 'Huang' or 'random'
n_clusters = 4                          # how many clusters (hyper parameter)
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
cluster_df = pd.DataFrame(columns=('categories', 'GoodForKids', 'stars',
                                   'RestaurantsPriceRange2',
                                   'latitude', 'longitude','cluster_id'))
#
#       load arrays back into a dataframe
#
for array in proto_cluster_assignments:
        cluster_df = cluster_df.append({'categories':array[0][0],'GoodForKids':array[0][1], 'stars':array[0][2],
                                    'RestaurantsPriceRange2':array[0][3],'latitude':array[0][4],'longitude':array[0][5],
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
plt.title('Estimated number of clusters: %d Epsilon:%.3f Min Samples:%d')
plt.show()



    
#Clustering
 
coords = restaurants_and_food.as_matrix(columns=['latitude', 'longitude'])
epsilon=0.005
minsamples=5
db = DBSCAN(eps=epsilon, min_samples=minsamples, algorithm='ball_tree', metric='euclidean').fit(coords)
cluster_labels = db.labels_

geo_df['cluster_labels'] = cluster_labels
num_clusters = len(set(cluster_labels))
clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
print('Number of clusters: {}'.format(num_clusters))
core_samples = db.core_sample_indices_
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True


#ploting
fig,ax = plt.subplots(figsize = (15,15))
#toronto_boundaries.plot(ax=ax, alpha=0.4, color="grey")
# Black removed and is used for noise instead.
unique_labels = set(cluster_labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (cluster_labels == k)

    xy = coords[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = coords[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d Epsilon:%.3f Min Samples:%d' % (num_clusters ,epsilon, minsamples))
plt.show()
  


