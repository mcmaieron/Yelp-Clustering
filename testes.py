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
import mca
from sklearn.preprocessing import normalize
from sklearn.metrics import  silhouette_score
from sklearn.cluster import KMeans
from scipy.spatial import distance
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import prince
import string
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import CountVectorizer
from scipy.cluster.hierarchy import dendrogram, linkage

#from fuzzywuzzy import fuzz
####preliminary analisys####

#Loads all business in business.json
business = pd.read_json(r'business.json', lines=True)
business.columns
# Get names of indexes for which column city has value different from Toronto
indexNames = business[ business['city'] != "Toronto" ].index
# Delete these row indexes from dataFrame
business.drop(indexNames , inplace=True)
# Reset the index after dropping rows
business.reset_index(drop=True, inplace=True)

# simply drop whole row with NaN in "attributes" column
No_Na= business.copy()
No_Na.dropna(subset=["attributes","categories"], axis=0, inplace=True)
# Reset the index after dropping rows
No_Na.reset_index(drop=True, inplace=True)

# Create a mask to select stablishments that have one of the following words 
# "Restaurants" or "Food" or "Bar"
mask_restaurants = No_Na['categories'].str.match('Restaurants|Food|Bars')

# apply mask
restaurants_and_food = No_Na[mask_restaurants]
# Reset the index after selection
restaurants_and_food.reset_index(drop=True, inplace=True)

#Create a copy and put words in alphabetical order to comparison
restaurants_and_food_j= restaurants_and_food.copy()
restaurants_and_food_j['categories'] = restaurants_and_food_j['categories'].apply(lambda x : ", ".join(sorted(x.split(","), key=lambda x: x.split())))


restaurants_and_food_j.categories = restaurants_and_food_j.categories.str.strip()
restaurants_and_food_j.categories = restaurants_and_food_j.categories.str.replace(",  ",", ")
restaurants_and_food_j.categories = restaurants_and_food_j.categories.str.replace(", Food","")
restaurants_and_food_j.categories = restaurants_and_food_j.categories.str.replace(", Food,",",")

for i in restaurants_and_food_j['categories']:
    if i.startswith('Food,')==True:
        restaurants_and_food_j.categories = restaurants_and_food_j.categories.str.replace("Food, ","")
    
    if 'Coffee & Tea' in i and 'Cafes' in i:
        if i.startswith('Cafes,'):
            restaurants_and_food_j.categories = restaurants_and_food_j.categories.str.replace("Cafes, ","")
        else: 
             restaurants_and_food_j.categories = restaurants_and_food_j.categories.str.replace("Cafes, ","")

restaurants_and_food_j['categories'].value_counts()
 
# Creates a set with all different categories. It is used to find categories 
# thas does not fit in the domain and may interfer in the clustering step
results = set()
restaurants_and_food_j['categories'].str.lower().str.split(',').apply(results.update)
results


categoriesToRemove = [
                        'Grocery','Venues & Event Spaces','Drugstores',
                        'Convenience Stores','Beauty & Spas',
                        'Photography Stores & Services','Cosmetics & Beauty Supply',
                        'Discount Store','Department Stores',
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
                        'Dance Schools','Performing Arts','Wholesale Stores',
                        'Tobacco Shops','Nutritionists','Hobby Shops',
                        'Pet Services','Electronics','Plumbing','Gyms',
                        'Yoga','Walking Tours','Toy Stores','Pet Stores',
                        'Pet Groomers','Vape Shops','Head Shops',
                        'Souvenir Shops','Pharmacy','Appliances & Repair',
                        'Wholesalers','Party Equipment Rentals','Tattoo',
                        'Funeral Services & Cemeteries','Sporting Goods',
                        'Dog Walkers','Pet Boarding/Pet Sitting',
                        'Scavenger Hunts','Contractors','Trainers',
                        'Customized Merchandise', 'Dry Cleaning & Laundry',
                        'Art Galleries','Tax Law', 'Bankruptcy Law',
                        'Tax Services', 'Estate Planning Law',
                        'Business Consulting', 'Lawyers', 'Pet Adoption',
                        'Escape Games', 'Animal Shelters','Fashion',
                        'Commercial Real Estate', 'Real Estate Agents',
                        'Real Estate Services', 'Home Inspectors',
                        'Activities','Antiques','Appliances','Arcades',
                        'Arts','Aooks','Aookstores','Building',
                        'Cards','Care','Casinos','Caterers','Centers',
                        'Churches','Classes','Computer','Education',
                        'Entertainment','Government','Instruments','Massage',
                        'Organizations','Schools','Stationery','Supplements',
                        'Supplies','Barbers','Hair Salons','Arts & Entertainment',
                        'Performing Arts','Venues & Event Spaces','Books',
                        'Vinyl Records','Mags','Butcher','Farmers Market',
                        'Hair Stylist','Shopping','Arts & Crafts','Knitting Supplies',
                        'Hair Removal','Nail Salons','Skin Care','Eyelash Service',
                        'Waxing','Local Services','Meat Shops','Pool Halls',
                        'Health Markets','Day Spas','Party & Event Planning',
                        'Cinema','Active Life','Dance Studios','Opera & Ballet',
                        'Pilates','Jazz & Blues','Flowers & Gifts','Gift Shops',
                        'Furniture Stores','Party Supplies','Makeup Artists',
                        'Kitchen Supplies','Floral Designers','Thrift Stores',
                        'Bike Rentals','Shared Office Spaces','Bikes Delivery Services',
                        'Banks','Outlet Stores','Marketing','Web Design','Life Coach',
                        'Reunion','Mini Golf','Golf','Drive-in Theater','Luggage',
                        'Mattresses','Video Game Stores','Tires','Poutineries',
                        'Fitness/Exercise Equipment','Lan Centers','Airports',
                        'Transportation','Sports Wear','Surf Schools','Swimwear',
                        'Beaches','Tanning','Cooking Schools','Psychic Mediums',
                        'Spiritual Shop','Stadiums & Arenas','Historical Tours',
                        'Photographers','Session Photography','Bowling','Acne Treatment',
                        'Laser Hair Removal','Medical Spas','Nail Technicians',
                        'Blow Dry/Out Services','Axe Throwing','Meditation Centers',
                        'Mobile Phone Repair','It Services & Computer Repair',
                        'Home Services','Skating Rinks','Tennis','Doctors','Donairs',
                        'Computers','Cannabis Clinics','Airlines','Acupuncture',
                        'Adult Entertainment','Accountants',"Women's Clothing",
                        'Videos & Video Game Rental','Sewing & Alterations', 'Shaved Ice',
                        'Shaved Snow','Shipping','Shoe Repair','Shoe Stores',
                        'Real Estate','Recreation','Religious','Printing Services',
                        'Professional Services','Public Markets','Playgrounds',
                        'Pet Boarding','Pet Sitting','Passport & Visa Services',
                        'Laundromat','Laundry Services','Leather Goods','Bookstores',
                        'Cards & Stationery','Community Centers','Community Service/Non-Profit',
                        'Couriers & Delivery Services','Ethical Grocery','Festivals',
                        'Financial Services','Florists','Hotels','Hotels & Travel',
                        'International Grocery','Medical Centers',"Men's Clothing",
                        'Motorcycle Dealers','Oganic Stores','Personal Shopping',
                        'Pets''Public Services & Government','Religious Organizations',
                        'Tabletop Games','Travel Sevices','Art Classes'
                        ]

# Remove the categories
#pat = r'\b(?:{})\b'.format('|'.join(categoriesToRemove))
#restaurants_and_food_j['categories'] = restaurants_and_food_j['categories'].str.replace(pat, '')
#restaurants_and_food_j['categories'] = restaurants_and_food_j['categories'].str.replace(', ,', ',')

# NEW CODE - CHANGED 13/11/2019
##########################################################################################################################################################################
import copy
def categories_cleanup(str, categoriesToRemove):
    '''
    Function created to take as input a string from the 'categories' column and output the same string without items from the 
    categoriesToRemove array.
    '''
    words_list = str.split(', ')
    words_list_copy = copy.deepcopy(words_list) #this copy is necessary because you can't delete items from a list you're iterating over
    for i in words_list:
        if i in categoriesToRemove:
            words_list_copy.remove(i)

    new_str = ", ".join(words_list_copy)
    # return the new string after removing the "bad" categories
    return new_str

# Removing categories that don't matter from the dataset, and entries that have no categories
restaurants_and_food_j_copy = copy.deepcopy(restaurants_and_food_j)

var = 0
for i in range(len(restaurants_and_food_j['categories'])):
    restaurants_and_food_j.iloc[i,restaurants_and_food_j.columns.get_loc('categories')] = categories_cleanup(restaurants_and_food_j['categories'][i], categoriesToRemove)
    if restaurants_and_food_j['categories'][i] != '':
        restaurants_and_food_j_copy.iloc[var] = restaurants_and_food_j.iloc[i]
        var += 1

restaurants_and_food_j = copy.deepcopy(restaurants_and_food_j_copy)
####################################################################################################################################


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
exp_rest = create_attributes(restaurants_and_food_j_copy,l)

## Drop columns that does not fit in the domain or that have the same value for 
## all instances
exp_rest.drop(['HairSpecializesIn','BusinessAcceptsCreditCards','BusinessAcceptsBitcoin','AgesAllowed'], axis=1,inplace=True)

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
        
# NEW CODE - CHANGED 04/12/2019
##########################################################################################################################################################################

# Calculates distance matrix of columns with numeric or boolean data

res = pdist(exp_rest[['stars']], 'euclidean')
dist_stars =squareform(res)
dist_stars =normalize(squareform(res),norm='max')

res = pdist(exp_rest[['RestaurantsPriceRange2']], 'euclidean')
dist_price=normalize(squareform(res),norm='max')

def calc_dist_bool (field):
    vect = CountVectorizer().fit_transform(field)
    vectors = vect.toarray() 
    dist = cosine_distances(vectors)
    return dist   

dist_bike = calc_dist_bool(exp_rest['BikeParking'])
dist_take =calc_dist_bool(exp_rest['RestaurantsTakeOut'])
dist_caters = calc_dist_bool(exp_rest['Caters'])
dist_reservations = calc_dist_bool(exp_rest['RestaurantsReservations'])
dist_kids = calc_dist_bool(exp_rest['GoodForKids'])
dist_out = calc_dist_bool(exp_rest['OutdoorSeating'])
dist_delivery = calc_dist_bool(exp_rest['RestaurantsDelivery']) 
dist_groups = calc_dist_bool(exp_rest['RestaurantsGoodForGroups'])
dist_wheel = calc_dist_bool(exp_rest['WheelchairAccessible'])
dist_dancing = calc_dist_bool(exp_rest['GoodForDancing'])
dist_dogs = calc_dist_bool(exp_rest['DogsAllowed'])
dist_tv = calc_dist_bool(exp_rest['HasTV'])
dist_table = calc_dist_bool(exp_rest['RestaurantsTableService'])
dist_drive = calc_dist_bool(exp_rest['DriveThru'])
dist_appo = calc_dist_bool(exp_rest['ByAppointmentOnly'])
dist_happy = calc_dist_bool(exp_rest['HappyHour'])
dist_coat = calc_dist_bool(exp_rest['CoatCheck'])

#w = len(exp_rest.index)
#dist = np.zeros(shape=(w,w))
#for j in exp_rest:
#    if j in [
#            'BikeParking','RestaurantsTakeOut',
#            'Caters', 'RestaurantsReservations',
#            'GoodForKids','OutdoorSeating','RestaurantsDelivery','HasTV',
#            'RestaurantsGoodForGroups','RestaurantsTableService','DogsAllowed',
#            'WheelchairAccessible','DriveThru','ByAppointmentOnly','GoodForDancing',
#            'HappyHour','CoatCheck'
#           ]:
#            print('coluna: ',j)
#            vect = CountVectorizer().fit_transform(exp_rest[j])
#            vectors = vect.toarray() 
#            dist = np.dstack((dist,cosine_distances(vectors)))


# Calculates distance matrix of text columns    
def clean_string (text):
    text = ''.join([word for word in text if word not in string.punctuation])
    text = text.lower()
    return text

def calc_dist_text (field):
    cleaned = (map(clean_string, field))
    vect = CountVectorizer().fit_transform(cleaned)
    vectors = vect.toarray() 
    dist = cosine_distances(vectors)
    return dist  

def calc_dist_text_vec (field):
    cleaned = (map(clean_string, field))
    vect = CountVectorizer().fit_transform(cleaned)
    vectors = vect.toarray() 
    return vectors  


dist_cat = calc_dist_text(exp_rest.categories)
dist_business = calc_dist_text(exp_rest.BusinessParking)
dist_att = calc_dist_text(exp_rest.RestaurantsAttire)
dist_alcohol = calc_dist_text(exp_rest.Alcohol)
dist_noise = calc_dist_text(exp_rest.NoiseLevel)
dist_wifi = calc_dist_text(exp_rest.WiFi)
dist_ambience = calc_dist_text(exp_rest.Ambience)
dist_smoking = calc_dist_text(exp_rest.Smoking)
dist_music = calc_dist_text(exp_rest.Music)
dist_nights = calc_dist_text(exp_rest.BestNights)
dist_meal = calc_dist_text(exp_rest.GoodForMeal)
dist_dietary = calc_dist_text(exp_rest.DietaryRestrictions)


# Sum all distance matrix and normalizes

dist =normalize((dist_alcohol  + dist_ambience + dist_nights + dist_dietary + dist_meal 
+ dist_nights + dist_music + dist_smoking + dist_wifi + dist_noise + dist_att 
+ dist_business + dist_cat + dist_coat + dist_happy+ dist_appo + dist_drive 
+ dist_table + dist_tv + dist_dogs + dist_dancing + dist_wheel + dist_groups  
+ dist_delivery + dist_out + dist_kids + dist_caters + dist_take
+ dist_bike + dist_price + dist_stars),norm='max')



## calculates silhoette coeficient in a max number of cluster and plots it
def evaluate_silhoette(dist, max_clusters):
    error = np.zeros(max_clusters+1)
    error[0] = 0;
    link = ['complete', 'average', 'single']
    for i in link:        
        for k in range(19,max_clusters+1):
           model2 = AgglomerativeClustering(linkage=i,n_clusters= k, affinity = 'precomputed')
           labels = model2.fit_predict(dist)
           error[k] = silhouette_score(dist, labels, metric='precomputed')
           print('N_cluster: ',k, 'Avg silhoette: ',error[k])
    fig, ax = plt.subplots()
    plt.title('Agglomerative Clustering')
    ax.plot(range(1,len(error)),error[1:],label=i)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhoette coeficient')
    plt.legend(title='Linkage criteria:')
    plt.show()

evaluate_silhoette(dist, 20)

#plot dendogram
def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


model = AgglomerativeClustering(linkage='average',n_clusters= 15, affinity = 'precomputed')

model = model.fit(dist)
fig,ax = plt.subplots(figsize = (45,45),dpi=400)
plt.title('Hierarchical Clustering Dendrogram')
plot_dendrogram(model, labels=model.labels_, orientation='left',truncate_mode='level',p=12)
plt.show()
plt.savefig(r'F:\Mestrado\Computacao\KDD\TrabalhoFinal\apresentacao\dendogramDist.pdf')  

# NEW CODE - CHANGED 13/11/2019
##########################################################################################################################################################################
## Reduce dimensions using MCA algorithm
dum = pd.get_dummies(exp_rest['categories'])
num_col = len(dum.columns)
mca_ben = mca.MCA(dum,ncols=num_col)
teste = (mca_ben.fs_r())
factor = mca_ben.fs_r(N=2).T
teste.L

factort = factor.T 
factort[:,0]
exp_rest['Fac1'] = factort[:,0].tolist()
exp_rest['Fac2'] = factort[:,1].tolist()




mca = prince.MCA(
     n_components=2,
     n_iter=3,
     copy=True,
     check_input=True,
     engine='auto',
     random_state=42
                 )
mca = mca.fit(dum)
mca.eigenvalues_
mca.total_inertia_

ax = mca.plot_coordinates(
    X=dum,
    ax=None,
    figsize=(6, 6),
    show_row_points=True,
    row_points_size=10,
    show_row_labels=False,
    show_column_points=True,
    column_points_size=30,
    show_column_labels=False,
    legend_n_cols=1
    )
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

fields.append('latitude')
fields.append('longitude')
fields.append('Fac1')
fields.append('Fac2')
#
#       select fields
columns_to_normalize     = ['latitude', 'longitude', 'Fac1', 'Fac2']
meanlat = exp_rest['latitude'].mean()
meanlong = exp_rest['longitude'].mean()
stdlat = np.std(exp_rest['latitude'])
stdlong = np.std(exp_rest['longitude'])
exp_rest[columns_to_normalize] = exp_rest[columns_to_normalize].apply(lambda x: (x - x.mean()) / np.std(x))

data_cats = exp_rest.loc[:,fields]
#
#       normalize continous fields
#
#       essentially compute the z-score
#
#       note: Could use (x.max() - x.min()) instead of np.std(x)
#

#
#       kprototypes needs an array
#
data_cats_matrix = data_cats.as_matrix()        

def evaluate_clusters(final_df,max_clusters):
    error = np.zeros(max_clusters+1)
    error[0] = 0;
    for k in range(1,max_clusters+1):
        kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
        kmeans.fit_predict(final_df)
        error[k] = kmeans.inertia_
    plt.figure(1)
    plt.plot(range(1,len(error)),error[1:])
    plt.xlabel('Number of clusters')
    plt.ylabel('Error')
    
    
#    def evaluate_clusters(final_df,max_clusters):
#    error = np.zeros(max_clusters+1)
#    error[0] = 0;
#    for k in range(1,max_clusters+1):
#        kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
#        kmeans.fit_predict(final_df)
#        labels = kmeans.labels_
#        print('fddf', labels)
#        error[k] =  silhouette_score(final_df, labels)
#    plt.figure(1)
#    plt.plot(range(1,len(error)),error[1:])
#    plt.xlabel('Number of clusters')
#    plt.ylabel('Error')
#       model parameters
#
evalu = data_cats.copy() 
evalu.drop(['GoodForKids'], axis=1,inplace=True)
evaluate_clusters(evalu,50)
init       = 'Huang'                    # init can be 'Cao', 'Huang' or 'random'
n_clusters =20                        # how many clusters (hyper parameter)
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
cluster_df = pd.DataFrame(columns=('GoodForKids', 'stars',
                                   'RestaurantsPriceRange2',
                                   'latitude', 'longitude','Fac1','Fac2','cluster_id'))
#
#       load arrays back into a dataframe
#
for array in proto_cluster_assignments:
        cluster_df = cluster_df.append({'GoodForKids':array[0][0], 'stars':array[0][1],
                                    'RestaurantsPriceRange2':array[0][2],'latitude':array[0][3],'longitude':array[0][4],
                                    'Fac1':array[0][5],'Fac2':array[0][6],'cluster_id':array[1]}, ignore_index=True)


cluster_df.cluster_id.value_counts()


summary = cluster_df.describe(include="all")
summary
summary.to_csv(r'F:\Mestrado\Computacao\KDD\TrabalhoFinal\dados\summary.csv')


cluster_df.RestaurantsPriceRange2 = cluster_df.RestaurantsPriceRange2.astype(float)

cluster_df.GoodForKids
d = {'True': True, 'False': False}
cluster_df.GoodForKids = cluster_df.GoodForKids.map(d)
cluster_df['GoodForKids'] =cluster_df['GoodForKids'].astype(int)


summary_cluster = cluster_df.groupby('cluster_id').describe(include="all")
summary_cluster
summary_cluster.to_csv(r'F:\Mestrado\Computacao\KDD\TrabalhoFinal\dados\summary_cluster50.csv')
##Ploting location## 
#loads US/Canada shapefile
world = gpd.read_file('F:\Mestrado\Computacao\KDD\TrabalhoFinal\dados\shape_mundo\World_Countries.shp')
toronto_boundaries = gpd.read_file(r'F:\Mestrado\Computacao\KDD\TrabalhoFinal\dados\toronto_boundaries\toronto_boundaries.shp')
#Plots the world shape
#world.plot(ax=ax, alpha=0.4, color="grey")
#Set CRS as 4326 (WGS84)
crs = {'init': 'epsg:4326'}    

# Create a list with the pair of coordinate (LONG,LAT)    
geometry = [Point(xy) for xy in zip (cluster_df["longitude"]*stdlong+meanlong,cluster_df["latitude"]*stdlat+meanlat)]
geometry
# Create a datafreme with the column geometry and the crs
geo_df = gpd.GeoDataFrame(cluster_df,crs=crs, geometry=geometry)

#Sets the size of the graph
fig,ax = plt.subplots(figsize = (15,15))

#Plots the world shape
#world.plot(ax=ax, alpha=0.4, color="grey")
toronto_boundaries.plot(ax=ax, alpha=0.4, color="grey")
#Plots the points
geo_df.plot(ax=ax, markersize=20)
plt.title('Algorithm: K-Means - Number of clusters: {}'.format(n_clusters))
plt.show()



print ('elem clus: ', cluster_df.RestaurantsPriceRange2[cluster_df['cluster_id'] == 5])
geo_df.to_file(driver = 'ESRI Shapefile', filename= "cluster_toronto.shp")
X = distance.cdist([cluster_df["longitude"], cluster_df["latitude"]],[cluster_df["longitude"], cluster_df["latitude"]] ,'euclidean')
silhouette_avg = silhouette_score(X,cluster_df['cluster_id'], metric="precomputed")
print("For n_clusters =",n_clusters,
          "The average silhouette_score is :", silhouette_avg)



# k-MEANS 
X = restaurants_and_food_j[['latitude', 'longitude']].values
kmean = KMeans(n_clusters=20).fit(X)
cluster_df['cluster_K'] = kmean.predict(X).tolist()
#cluster_df['cluster_K'].value_counts()
summary_kmean = cluster_df.groupby('cluster_K').describe(include="all")
summary_kmean
summary_kmean.to_csv(r'F:\Mestrado\Computacao\KDD\TrabalhoFinal\dados\summary_kmean50.csv')
