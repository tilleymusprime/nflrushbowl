#!/usr/bin/env python
# coding: utf-8

# In[1]:


#First, we will import pandas and numpy
import pandas as pd
import numpy as np

#Next, we will import our dataset
df = pd.read_csv('C://Users//tilleymusprime//Desktop//train.csv')

#Let's take a look at the columns and shape of the data

print(df.columns)
print(df.shape)


# In[2]:


#Next, we will need to make some adjustments to the possession team column so that all the teams have the same abbreviation
#for every variable

df['PossessionTeam'].replace('BLT', 'BAL', inplace=True)
df['PossessionTeam'].replace('CLV', 'CLE', inplace=True)
df['PossessionTeam'].replace('ARZ', 'ARI', inplace=True)
df['PossessionTeam'].replace('HST', 'HOU', inplace=True)

#Now, we will create two columns. One will show the abbreviation for the team currently on offense.
#The second column will show the abbreviation for the team that is currently on defense.
home_abbr = list(df['HomeTeamAbbr'])
visitor_abbr = list(df['VisitorTeamAbbr'])
possession = list(df['PossessionTeam'])
offense = []
defense = []
for h,i,j in zip(range(len(home_abbr)), range(len(visitor_abbr)), range(len(possession))):
    if possession[j] == home_abbr[h]:
        offense.append(home_abbr[h])
        defense.append(visitor_abbr[i])
    elif possession[j] == visitor_abbr[i]:
        offense.append(visitor_abbr[i])
        defense.append(home_abbr[h])
df['offense'] = offense
df['defense'] = defense


#Next, we will create a column that shows what team the player is on
team = list(df['Team'])
tea = []
for h, i, j in zip(range(len(team)), range(len(home_abbr)), range(len(visitor_abbr))):
    if team[h] == 'away':
        tea.append(visitor_abbr[j])
    elif team[h] == 'home':
        tea.append(home_abbr[i])

df['team'] = tea

#Next, we will create a column that shows if a player is on offense or defense
od = []
for h, i, j in zip(range(len(offense)), range(len(tea)), range(len(defense))):
    if tea[i] == offense[h]:
        od.append('offense')
    elif tea[i] == defense[j]:
        od.append('defense')
print(len(od))
        
df['offense/defense'] = od

#Our next column will show how many yards there are to the endzone
yte = []
yardline = list(df['YardLine'])
fieldposition = list(df['FieldPosition'])
for h,i, j in zip(range(len(yardline)), range(len(fieldposition)), range(len(possession))):
    if possession[j] == fieldposition[i]:
        yte.append((50-yardline[h]) + 50)
    elif possession[j] != fieldposition[i]:
        yte.append(yardline[h])
    elif yardline[h] == 50:
        yte.append(50)
df['YardsToEndzone'] = yte

#Next we will combine the quarter and gameclock column to show how much time is left in the game

gameclock = list(df['GameClock'])
time_left = []
gc = []
#a = []
for i in range(len(gameclock)):
    
    a = gameclock[i].split(':')
    minutes = (int(a[0])*60)
    seconds = int(a[1])
    gc.append(minutes + seconds)
    
gc
df['secondsinquarter'] = gc

down = list(df['Down'])
quarter = list(df['Quarter'])
dr = []
for i in range(len(down)):
    if down[i] ==1:
        dr.append(4)
    elif down[i] == 2:
        dr.append(3)
    elif down[i] == 3:
        dr.append(2)
    elif down[i] == 4:
        dr.append(1)

        
qr = []
for i in range(len(quarter)):
    if quarter[i] ==1:
        qr.append(4)
    elif quarter[i] == 2:
        qr.append(3)
    elif quarter[i] == 3:
        qr.append(2)
    elif quarter[i] == 4:
        qr.append(1)
    else:
        qr.append(1)
        
df['downsremaining'] = dr
df['quartersremaining'] = qr

timeremaining = []
for h, i in zip(range(len(gc)), range(len(qr))):
    timeremaining.append(gc[h] * qr[i])
df['Secondsremaining'] = timeremaining

#Our next column will combine down and distance into a column of yards until first down / # of plays remaining
distance = list(df['Distance'])
y2g = []
for h, i in zip(range(len(dr)), range(len(distance))):
    y2g.append(distance[i] / dr[h])
df['YardstoFirstPerPlay'] = y2g

#Next, we will create a column for total points scored and another column for the score differential from the perspective of the
#offense
df['Total Points'] = df['HomeScoreBeforePlay'] + df['VisitorScoreBeforePlay']

hsbp = list(df['HomeScoreBeforePlay'])
vsbp = list(df['VisitorScoreBeforePlay'])
poss = list(df['PossessionTeam'])
ha = list(df['HomeTeamAbbr'])
va = list(df['VisitorTeamAbbr'])
sd = []
for h, i, j, k, l in zip(range(len(hsbp)), range(len(vsbp)), range(len(ha)), range(len(va)), range(len(poss))):
    if poss[l] == ha[j]:
        sd.append(hsbp[h] - vsbp[i])
    elif poss[l] == va[k]:
        sd.append(vsbp[i] - hsbp[h])
    else:
        sd.append('Error')
df['ScoreDifferential'] = sd

#Next, we will create a column to indicate if a player is the ball carrier or not. 
rusher = []
nflid = list(df['NflId'])
rushid = list(df['NflIdRusher'])
for h, i in zip(range(len(nflid)), range(len(rushid))):
    if nflid[h] == rushid[i]:
        rusher.append(1)
    elif nflid[h] != rushid[i]:
        rusher.append(0)
df['Rusher'] = rusher


# In[3]:


#Next, we will look at stadiumtype and condense that variable down to 2 entries (outdoor and indoor)
df['StadiumType'].replace('Outdoors', 'Outdoor', inplace=True)
df['StadiumType'].replace('Open', 'Outdoor', inplace=True)
df['StadiumType'].replace('Oudoor', 'Outdoor', inplace=True)
df['StadiumType'].replace('Open', 'Outdoor', inplace=True)
df['StadiumType'].replace('Retr. Roof-Open', 'Outdoor', inplace=True)
df['StadiumType'].replace('Bowl', 'Outdoor', inplace=True)
df['StadiumType'].replace('Outddors', 'Outdoor', inplace=True)
df['StadiumType'].replace('Heinz Field', 'Outdoor', inplace=True)
df['StadiumType'].replace('Retr. Roof - Open', 'Outdoor', inplace=True)
df['StadiumType'].replace('Bowl', 'Outdoor', inplace=True)
df['StadiumType'].replace('Outdoor Retr Roof-Open', 'Outdoor', inplace=True)
df['StadiumType'].replace('Ourdoor', 'Outdoor', inplace=True)
df['StadiumType'].replace('Outdor', 'Outdoor', inplace=True)
df['StadiumType'].replace('Indoor, Open Roof', 'Outdoor', inplace=True)
df['StadiumType'].replace('Outside', 'Outdoor', inplace=True)
df['StadiumType'].replace('Domed, Open', 'Outdoor', inplace=True)
df['StadiumType'].replace('Domed, open', 'Outdoor', inplace=True)
df['StadiumType'].replace('Cloudy', 'Outdoor', inplace=True)


df['StadiumType'].replace('Indoors', 'Indoor', inplace=True)
df['StadiumType'].replace('Dome', 'Indoor', inplace=True)
df['StadiumType'].replace('Retractable Roof', 'Indoor', inplace=True)
df['StadiumType'].replace('Retr. Roof-Closed', 'Indoor', inplace=True)
df['StadiumType'].replace('Retr. Roof - Closed', 'Indoor', inplace=True)
df['StadiumType'].replace('Domed, closed', 'Indoor', inplace=True)
df['StadiumType'].replace('Closed Dome', 'Indoor', inplace=True)
df['StadiumType'].replace('Dome, closed', 'Indoor', inplace=True)
df['StadiumType'].replace('Domed', 'Indoor', inplace=True)
df['StadiumType'].replace('Indoor, Roof Closed', 'Indoor', inplace=True)
df['StadiumType'].replace('Retr. Roof Closed', 'Indoor', inplace=True)

#We will do a similar process with the Turf column
#Due to a lack of expertise in grass knowledge, we will keep the specialty surface labels the same (ie a turf titan
#will stay as turf titan instead of changing to turf)
df['Turf'].replace('Natural Grass', 'Grass', inplace=True)
df['Turf'].replace('Naturall Grass', 'Grass', inplace=True)
df['Turf'].replace('Natural grass', 'Grass', inplace=True)
df['Turf'].replace('Natural', 'Grass', inplace=True)
df['Turf'].replace('Naturall Grass', 'Grass', inplace=True)
df['Turf'].replace('grass', 'Grass', inplace=True)
df['Turf'].replace('natural grass', 'Grass', inplace=True)

df['Turf'].replace('Field Turf', 'Turf', inplace=True)
df['Turf'].replace('Artificial', 'Turf', inplace=True)
df['Turf'].replace('FieldTurf', 'Turf', inplace=True)
df['Turf'].replace('Artifical', 'Turf', inplace=True)
df['Turf'].replace('Field turf', 'Turf', inplace=True)

#Now that we have stadium type edited, we will fill in the missing values of temperature with 75 if it is an indoor stadium
df['Temperature'].fillna('Unknown', inplace=True)
temp = list(df['Temperature'])
stadium = list(df['StadiumType'])

tem = []
for h, i in zip(range(len(temp)), range(len(stadium))):
    if temp[h] != 'Unknown':
        tem.append(temp[h])
    elif ((temp[h] == 'Unknown') & (stadium[i] == 'Outdoor')):
        tem.append('Unknown Outdoors')
    elif ((temp[h] == 'Unknown') & (stadium[i] == 'Indoor')):
        tem.append(75)
    else:
        tem.append('Error')
df['temperature'] = tem
df['temperature'].value_counts()
#It looks like our outdoor unknown comes from one game in Atlanta. Since Atlanta isn't known for its cold weather we will
#treat it like an indoor stadium
df['temperature'].replace('Unknown Outdoors', 75, inplace = True)


# In[4]:


df['GameWeather'].fillna('Unknown', inplace=True)
df['GameWeather'].replace('Partly Cloudy', 'Cloudy', inplace=True)
df['GameWeather'].replace('Party Cloudy', 'Cloudy', inplace=True)
df['GameWeather'].replace('Mostly Cloudy', 'Cloudy', inplace=True)
df['GameWeather'].replace('Mostly cloudy', 'Cloudy', inplace=True) 
df['GameWeather'].replace('Partly cloudy', 'Cloudy', inplace=True)
df['GameWeather'].replace('cloudy', 'Cloudy', inplace=True)
df['GameWeather'].replace('Cloudy and Cool', 'Cloudy', inplace=True)
df['GameWeather'].replace('Mostly Coudy', 'Cloudy', inplace=True)
df['GameWeather'].replace('Cloudy, fog started developing in 2nd quarter', 'Cloudy', inplace=True)
df['GameWeather'].replace('Partly Clouidy', 'Cloudy', inplace=True)
df['GameWeather'].replace('Cloudy and cold', 'Cloudy', inplace=True)
df['GameWeather'].replace('Coudy', 'Cloudy', inplace=True)
df['GameWeather'].replace('Cloudy, chance of rain', 'Cloudy', inplace=True)


df['GameWeather'].replace('Sunny', 'Clear', inplace=True)
df['GameWeather'].replace('Mostly Sunny', 'Clear', inplace=True)
df['GameWeather'].replace('Partly Sunny', 'Clear', inplace=True)
df['GameWeather'].replace('Fair', 'Clear', inplace=True)
df['GameWeather'].replace('Sunny', 'Clear', inplace=True)
df['GameWeather'].replace('Partly sunny', 'Clear', inplace=True)
df['GameWeather'].replace('Overcast', 'Clear', inplace=True)
df['GameWeather'].replace('Clear skies', 'Clear', inplace=True)
df['GameWeather'].replace('Clear Skies', 'Clear', inplace=True)
df['GameWeather'].replace('Clear and cold', 'Clear', inplace=True)
df['GameWeather'].replace('Mostly sunny', 'Clear', inplace=True)
df['GameWeather'].replace('Sunny and warm', 'Clear', inplace=True)
df['GameWeather'].replace('Clear and warm', 'Clear', inplace=True)
df['GameWeather'].replace('Cold', 'Clear', inplace=True)
df['GameWeather'].replace('Mostly Sunny Skies', 'Clear', inplace=True)
df['GameWeather'].replace('Sunny, highs to upper 80s', 'Clear', inplace=True)
df['GameWeather'].replace('Clear and Sunny', 'Clear', inplace=True)
df['GameWeather'].replace('Clear and Cool', 'Clear', inplace=True)
df['GameWeather'].replace('Clear and sunny', 'Clear', inplace=True)
df['GameWeather'].replace('Sunny Skies', 'Clear', inplace=True)
df['GameWeather'].replace('Sunny and cold', 'Clear', inplace=True)
df['GameWeather'].replace('Sun & clouds', 'Clear', inplace=True)
df['GameWeather'].replace('Sunny and clear', 'Clear', inplace=True)
df['GameWeather'].replace('Sunny, Windy', 'Clear', inplace=True)
df['GameWeather'].replace('T: 51; H: 55; W: NW 10 mph', 'Clear', inplace=True)
df['GameWeather'].replace('Partly clear', 'Clear', inplace=True)

df['GameWeather'].replace('Light Rain', 'Rain', inplace=True)
df['GameWeather'].replace('Rain shower', 'Rain', inplace=True)
df['GameWeather'].replace('Rainy', 'Rain', inplace=True)
df['GameWeather'].replace('30% Chance of Rain', 'Rain', inplace=True)
df['GameWeather'].replace('Scattered Showers', 'Rain', inplace=True)
df['GameWeather'].replace('Cloudy, Rain', 'Rain', inplace=True)
df['GameWeather'].replace('Rain likely, temps in low 40s.', 'Rain', inplace=True)
df['GameWeather'].replace('Cloudy, 50% change of rain', 'Rain', inplace=True)
df['GameWeather'].replace('Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.', 'Rain', inplace=True)
df['GameWeather'].replace('Showers', 'Rain', inplace=True)
df['GameWeather'].replace('Rain Chance 40%', 'Rain', inplace=True)

df['GameWeather'].replace('Heavy lake effect snow', 'Snow', inplace=True)
df['GameWeather'].replace('Heavy lake effect snow', 'Snow', inplace=True)
df['GameWeather'].replace('Cloudy, light snow accumulating 1-3"', 'Snow', inplace=True)


df['GameWeather'].replace('Controlled Climate', 'Indoor', inplace=True)
df['GameWeather'].replace('N/A (Indoors)', 'Indoor', inplace=True)
df['GameWeather'].replace('Indoors', 'Indoor', inplace=True)
df['GameWeather'].replace('N/A Indoor', 'Indoor', inplace=True)


# In[5]:


#Next, we will create a column of lists that shows offense/defense combined with position
#We will group by the offense/defense column and position
df['o_d_pos'] = df['offense/defense'] + ' ' + df['Position']
a = pd.DataFrame(df.groupby('PlayId')['o_d_pos'].apply(list))
a
rush = df[df['Rusher'] == 1]
df1 = pd.merge(rush, a, left_on='PlayId', right_index=True)


# In[6]:


odpos = list(df1['o_d_pos_y'])
defense = []
offense = []
for h in range(len(odpos)):
    d = []
    o = []
    for i in range(len(odpos[h])):
        if odpos[h][i][0] == 'd':
            d.append(odpos[h][i][-2] + odpos[h][i][-1])
        elif odpos[h][i][0] == 'o':
            o.append(odpos[h][i][-2] + odpos[h][i][-1])
    defense.append(d)
    offense.append(o)
df1['Offensive Personnel'] = offense
df1['Defensive Personnel'] = defense


# In[7]:


off1 = []
oline = []
terb = []
for h in range(len(offense)):
    ol = []
    rb = []
    wr = []
    qb = []
    te = []
    mystery = []
    for i in range(len(offense[h])):
        if ((offense[h][i] == ' T') | (offense[h][i] == ' G') | (offense[h][i] == ' C') | (offense[h][i] == 'DT')):
            ol.append(offense[h][i])
        elif ((offense[h][i] == 'DE') | (offense[h][i] == 'OT') | (offense[h][i] == 'OG') | (offense[h][i] == 'NT')):
            ol.append(offense[h][i])
        elif ((offense[h][i] == 'WR') | (offense[h][i] == 'FS')):
            wr.append(offense[h][i])
        elif ((offense[h][i] == 'RB') | (offense[h][i] == 'HB')):
            rb.append(offense[h][i])
        elif offense[h][i] == 'QB':
            qb.append(offense[h][i])
        elif ((offense[h][i] == 'TE') | (offense[h][i] == 'FB')):
            te.append(offense[h][i])
        else:
            mystery.append(offense[h][i])
    off1.append('QB:' + str(len(qb)) + ' RB:' + str(len(rb)) + ' WR: '+ str(len(wr)) + ' TE:' + str(len(te)) + ' OL:' + str(len(ol)))
    oline.append(len(ol))
    terb.append(len(rb) + len(te) + len(qb))
df1['Off1'] = off1
df1['Oline'] = oline
df1['TERBQB'] = terb


# In[8]:


def1 = []
dline = []
linebackers = []
for h in range(len(defense)):
    dl = []
    lb = []
    secondary = []
    mystery = []
    for i in range(len(defense[h])):
        if ((defense[h][i] == ' T') | (defense[h][i] == ' G') | (defense[h][i] == ' C') | (defense[h][i] == 'DT')):
            dl.append(defense[h][i])
        elif ((defense[h][i] == 'DE') | (defense[h][i] == 'OT') | (defense[h][i] == 'OG') | (defense[h][i] == 'NT') |
              (defense[h][i] == 'DL')):
            dl.append(defense[h][i])
        elif ((defense[h][i] == 'WR') | (defense[h][i] == 'FS') | (defense[h][i] == 'SS') | (defense[h][i] == ' S')):
            secondary.append(defense[h][i])
        elif ((defense[h][i] == 'CB') | (defense[h][i] == 'DB') | (defense[h][i] == 'AF')):
            secondary.append(defense[h][i])
        elif ((defense[h][i] == 'LB') | (defense[h][i] == 'FB')):
            lb.append(defense[h][i])
        else:
            mystery.append(defense[h][i])
    def1.append('DL:' + str(len(dl)) + ' LB:' + str(len(lb)) + ' Secondary: '+ str(len(secondary)))
    dline.append(len(dl))
    linebackers.append(len(lb))
df1['Def1'] = def1
df1['Dline'] = dline
df1['Linebackers'] = linebackers


# In[9]:


df1['Box Players'] = (df1['TERBQB'] + df1['Oline']) - (df1['Dline'] + df1['Linebackers'])


# In[10]:


df.columns


# In[11]:


df = df1[['Yards', 'YardsToEndzone', 'Secondsremaining', 'YardstoFirstPerPlay',
       'ScoreDifferential', 'Oline', 'Dline', 'Linebackers', 'TERBQB']]


# In[ ]:





# In[12]:


import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
features = df.iloc[:, 1:]
labels = np.array(df['Yards'])
feature_list = list(features.columns)


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)


# In[14]:


rf = RandomForestRegressor(n_estimators =1000, random_state=42)
rf.fit(X_train, y_train)
importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x:x[1], reverse=True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# In[15]:


x = np.array(rf.predict(X_test))
a = pd.DataFrame([list(x), list(y_test)])
b = a.transpose()
b.columns = ['RFPredictions', 'Actual']
b['RFDifference'] = abs(b['RFPredictions'] - b['Actual'])
meanyards = []
for h in range(len(b)):
    meanyards.append(round(np.mean(df['Yards']), 4))
b['MeanGuess'] = meanyards
b['GuessActual'] = abs(b['MeanGuess'] - b['Actual'])
b.describe()


# In[ ]:





# In[16]:


from sklearn import svm


# In[17]:


svm = svm.SVR()
svm.fit(X_train, y_train)
s = np.array(svm.predict(X_test))
b['SVMPredictions'] = s
b['SVMDifference'] = abs(b['SVMPredictions'] - b['Actual'])


# In[ ]:





# In[18]:


b['SVMDifference'] = abs(b['SVMPredictions'] - b['Actual'])


# In[19]:


b.describe()


# In[21]:


b.to_csv('C://Users//tilleymusprime//Desktop//nflrfsvm.csv')


# In[24]:


X_test.shape


# In[25]:


X_test


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[8]:


##Variables defined in list form:
#home_abbr:home team abbreviation
#visitor_abbr: visitor team abbreviation
#possession: team currently on offense (by abbreviation)
#offense:
#defense:
#tea: team a player is on by abbreviation
#team: home/away designation

