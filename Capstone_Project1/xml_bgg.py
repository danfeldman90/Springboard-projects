import urllib.request as urlreq
from xml.etree import ElementTree as ET
import pandas as pd
import mpld3, time
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#import pdb

# Initialize empty lists to be filled with relevant board game data:
names = []
id_nums = []
ranks = []
weight = []
num_users = []
min_play = []
max_play = []
year_pub = []
min_players = []
max_players = []
std = []
category = {}
#
# Last stopped at:
# 'The program has successfully called up to ID 54200 so far of 120000!'
# File "<string>", line unknown
# ParseError: not well-formed (invalid token): line 1, column 9

# Loop through the API and make all the requests:
num = 210001
while num <= 250000:
    # Parameters defined here (ID numbers and yes to statistics):
    ids = ','.join([str(i) for i in range(num,num+100)])
    num += 100
    stats = '1'
    
    # Construct the URL using the rootpath, base of API, then params:
    rootpath = 'https://www.boardgamegeek.com/xmlapi/'
    base = '/boardgame/'
    requestURL = rootpath + base \
               + ids \
               + '?stats=' + stats
    
    # Request the xml and parse what's returned:
    root = ET.parse(urlreq.urlopen(requestURL)).getroot()
    
    items = root.findall('boardgame')
    for item in items:
        try:
            names.append(str(item.find('name').text))
            ranks.append(item.find('statistics').find('ratings').find('bayesaverage').text)
            weight.append(item.find('statistics').find('ratings').find('averageweight').text)
            num_users.append(item.find('statistics').find('ratings').find('usersrated').text)
            min_play.append(item.find('minplaytime').text)
            max_play.append(item.find('maxplaytime').text)
            year_pub.append(item.find('yearpublished').text)
            min_players.append(item.find('minplayers').text)
            max_players.append(item.find('maxplayers').text)
            std.append(item.find('statistics').find('ratings').find('stddev').text)
            idNum = item.attrib['objectid']
            id_nums.append(int(idNum))
            category[idNum] = []
            for categories in item.findall('boardgamecategory'):
                category[idNum].append(str(categories.text))
        except AttributeError:
            continue
    print('The program has successfully called up to ID '+ str(num-1) + ' so far of 250000!')
    time.sleep(5.5)

# Create a DataFrame using the retrieved data:
params = list(zip(id_nums,ranks,std,weight,num_users,min_play,max_play,year_pub,min_players,max_players))
gameDF = pd.DataFrame(index=names, data=params, columns=['ID Num', 'Rating', 'Standard Dev',\
                                            'Weight', 'User Number', 'Min Playtime', \
                                            'Max Playtime', 'Year Published', \
                                            'Min Players', 'Max Players'])
# Convert certain columns to float:
gameDF['Rating'] = gameDF['Rating'].astype(float)
gameDF['Standard Dev'] = gameDF['Standard Dev'].astype(float)
gameDF['Weight'] = gameDF['Weight'].astype(float)
#gameDF['User Number'] = gameDF['User Number'].astype(int)
#gameDF['Year Published'] = gameDF['Year Published'].astype(int)
#gameDF['Min Players'] = gameDF['Min Players'].astype(int)
#gameDF['Max Players'] = gameDF['Max Players'].astype(int)

for row, val in enumerate(gameDF['Year Published']):
    if val == None:
        gameDF['Year Published'][row] = 10000 #Arbitrarily high number placeholder
    elif int(val) < 0:
        gameDF['Year Published'][row] = 10000
gameDF['Year Published'] = gameDF['Year Published'].astype(float)
for row, val in enumerate(gameDF['Year Published']):
    if val == 10000:
        gameDF['Year Published'][row] = np.nan

for row, val in enumerate(gameDF['User Number']):
    if val == None:
        gameDF['User Number'][row] = 10000. #Arbitrarily high number placeholder
gameDF['User Number'] = gameDF['User Number'].astype(float)
for row, val in enumerate(gameDF['User Number']):
    if val == 10000.:
        gameDF['User Number'][row] = np.nan

for row, val in enumerate(gameDF['Min Playtime']):
    if val == None:
        gameDF['Min Playtime'][row] = 10000. #Arbitrarily high number placeholder
gameDF['Min Playtime'] = gameDF['Min Playtime'].astype(float)
for row, val in enumerate(gameDF['Min Playtime']):
    if val == 10000.:
        gameDF['Min Playtime'][row] = np.nan

for row, val in enumerate(gameDF['Max Playtime']):
    if val == None:
        gameDF['Max Playtime'][row] = 10000 #Arbitrarily high number placeholder
gameDF['Max Playtime'] = gameDF['Max Playtime'].astype(float)
for row, val in enumerate(gameDF['Max Playtime']):
    if val == 10000.:
        gameDF['Max Playtime'][row] = np.nan

for row, val in enumerate(gameDF['Min Players']):
    if val == None:
        gameDF['Min Players'][row] = 10000 #Arbitrarily high number placeholder
gameDF['Min Players'] = gameDF['Min Players'].astype(float)
for row, val in enumerate(gameDF['Min Players']):
    if val == 10000.:
        gameDF['Min Players'][row] = np.nan

for row, val in enumerate(gameDF['Max Players']):
    if val == None:
        gameDF['Max Players'][row] = 10000 #Arbitrarily high number placeholder
gameDF['Max Players'] = gameDF['Max Players'].astype(float)
for row, val in enumerate(gameDF['Max Players']):
    if val == 10000.:
        gameDF['Max Players'][row] = np.nan

# Mask entries with ratings of zero:
for ind, rate in enumerate(gameDF['Rating']):
    if rate == 0.0:
        gameDF['Rating'][ind] = np.nan

# Create a scatter plot of the data with tooltips (labels w/ game names):
# fig = plt.gcf()
# gameDF.plot(kind='scatter', x='Weight', y='Rating')
# plt.tight_layout()
# plt.title('Weight vs. Board Game Rating')
# labels = [str(name) for name in gameDF['Name']]
# ax = plt.gca()
# pts = ax.get_children()[3]
# tooltip = mpld3.plugins.PointLabelTooltip(pts, labels=labels)
# mpld3.plugins.connect(fig, tooltip)
# mpld3.display(fig)





