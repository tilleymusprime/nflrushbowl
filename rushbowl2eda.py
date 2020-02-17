#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.stats import pearsonr

sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


#


# In[5]:


df = pd.read_csv('C://Users//tilleymusprime//Desktop//df1.csv')


# In[6]:


#First, let's check to see if there are any variables significantly correlated with yardage.
dfcorr = df.corr()
dfcorr['Yards']
#It doesn't look like there are any significant correlations with yards


# In[8]:


# Let's take a look at the difference between the nickel, 3-4 and 4-3 defense. 
#The nickel defense will have 4 linemen, 5 defensive backs, and 2 linebackers
#The 3-4 has three linemen, 4 linebackers, and 4 defensive backs
#The 4-3 has 4 linemen, 3 linebackers, and four defensive backs.
formation = df[['Yards', 'DefensePersonnel', 'Down', 'Distance', 'defense']]
nickel = formation[formation['DefensePersonnel'] == '4 DL, 2 LB, 5 DB']
def43 = formation[formation['DefensePersonnel'] == '4 DL, 3 LB, 4 DB']
def34 = formation[formation['DefensePersonnel'] == '3 DL, 4 LB, 4 DB']
print(nickel['Yards'].describe(), def43['Yards'].describe(), def34['Yards'].describe())


# In[9]:


#Next, we will make histograms for the different defenses
plt.hist(nickel['Yards'], log=True)
plt.xlabel('Yards')
plt.ylabel('Times Occured')
plt.title('Nickel')
plt.savefig('C://Users//tilleymusprime//Desktop//nickel_full.png')
plt.show()


# In[10]:


plt.hist(def43['Yards'], log=True)
plt.xlabel('Yards')
plt.ylabel('Times Occured')
plt.title('4-3 Defense')
plt.savefig('C://Users//tilleymusprime//Desktop//43_full.png')
plt.show()


# In[11]:


plt.hist(def34['Yards'], log=True)
plt.xlabel('Yards')
plt.ylabel('Times Occured')
plt.title('3-4 Defense')
plt.savefig('C://Users//tilleymusprime//Desktop//34_full.png')
plt.show()


# In[12]:


#The yardage is much more heavily centered around 0 to 10 yards so we will remove the big plays from the data and look again
print(nickel.shape)
nickel1=nickel[((nickel['Yards'] >=-2) & (nickel['Yards']<=10))]
print(nickel1.shape)
print(def43.shape)
def431=def43[((def43['Yards'] >=-2) & (def43['Yards']<=10))]
print(def431.shape)
print(def34.shape)
def341=def34[((def34['Yards'] >=-2) & (def34['Yards']<=10))]
print(def341.shape)


# In[13]:


#Let's take a look at the statistical breakdown for each defense
print(nickel1['Yards'].describe(), def431['Yards'].describe(), def341['Yards'].describe())


# In[14]:


plt.hist(nickel1['Yards'], log=True)
plt.xlabel('Yards')
plt.ylabel('Times Occured')
plt.title('Nickel')
plt.savefig('C://Users//tilleymusprime//Desktop//nickel_nooutliers.png')
plt.show()


# In[15]:


plt.hist(def431['Yards'], log=True)
plt.xlabel('Yards')
plt.ylabel('Times Occured')
plt.title('4-3 Defense')
plt.savefig('C://Users//tilleymusprime//Desktop//43_nooutliers.png')
plt.show()


# In[16]:


plt.hist(def341['Yards'], log=True)
plt.xlabel('Yards')
plt.ylabel('Times Occured')
plt.title('3-4 Defense')
plt.savefig('C://Users//tilleymusprime//Desktop//34_nooutliers.png')
plt.show()


# In[ ]:





# In[17]:


df['LBDL'] = df['Dline'] + df['Linebackers']
pearsonr(df['Yards'], df['LBDL'])


# In[18]:


#Next, let's take a look at what formation is best on 3/4th down and short.
#We will only focus on third and fourth down where the yardage is 3 or less
ytg = df[(((df['Down'] == 3) | (df['Down'] == 4)) & (df['Distance'] <=3))]
ytg.shape


# In[19]:


off1 = list(ytg['Off1'])
wr = []
rb = []
te = []
for i in range(len(off1)):
    wr.append(int(off1[i][14]))
    rb.append(int(off1[i][8]))
    te.append(int(off1[i][-6]))
ytg['WR'] = wr
ytg['RB'] = rb
ytg['TE'] = te


# In[20]:


print('Offensive Linemen: ', pearsonr(ytg['Yards'], ytg['Oline']))
print('Wide Receivers: ',pearsonr(ytg['Yards'], ytg['WR']))
print('Tight Ends: ',pearsonr(ytg['Yards'], ytg['TE']))
print('Running Backs: ',pearsonr(ytg['Yards'], ytg['RB']))
#It looks like there is correlation between yards and receivers and yards and tight ends.
#Let's explore those two positions further


# In[21]:


#As we will see in the below data and histograms, it is actually easier to run the ball when there are more receivers on the
#field. 
ytg = ytg[ytg['Yards'] <=10]
zero = ytg[ytg['WR'] == 0]
print(zero['Yards'].describe())
one = ytg[ytg['WR'] == 1]
print(one['Yards'].describe())
two = ytg[ytg['WR'] == 2]
print(two['Yards'].describe())
three = ytg[ytg['WR'] == 3]
print(three['Yards'].describe())
four = ytg[ytg['WR'] == 4]
print(four['Yards'].describe())
five = ytg[ytg['WR'] == 5]
print(five['Yards'].describe())


# In[22]:


plt.hist(zero['Yards'])
plt.xlabel('Yards')
plt.ylabel('Times Occured')
plt.title('No Receivers')
plt.savefig('C://Users//tilleymusprime//Desktop//NoReceivers.png')
plt.show()


# In[23]:


plt.hist(one['Yards'])
plt.xlabel('Yards')
plt.ylabel('Times Occured')
plt.title('One Receiver')
plt.savefig('C://Users//tilleymusprime//Desktop//OneReceiver.png')
plt.show()


# In[24]:


plt.hist(two['Yards'])
plt.xlabel('Yards')
plt.ylabel('Times Occured')
plt.title('Two Receivers')
plt.savefig('C://Users//tilleymusprime//Desktop//TwoReceivers.png')
plt.show()


# In[25]:


plt.hist(three['Yards'])
plt.xlabel('Yards')
plt.ylabel('Times Occured')
plt.title('Three Receivers')
plt.savefig('C://Users//tilleymusprime//Desktop//ThreeReceivers.png')
plt.show()


# In[26]:


plt.hist(four['Yards'])
plt.xlabel('Yards')
plt.ylabel('Times Occured')
plt.title('Four Receivers')
plt.savefig('C://Users//tilleymusprime//Desktop//FourReceivers.png')
plt.show()


# In[ ]:





# In[27]:


##Unlike receviers, having more tight ends does not mean more yards (even though hypothetically they are receivers that can
#block.)Having 1 tight end can be helpful but haivng more than 1 leads to a lower average of yards.
ytg = ytg[ytg['Yards'] <=10]
zero = ytg[ytg['TE'] == 0]
print(zero['Yards'].describe())
one = ytg[ytg['TE'] == 1]
print(one['Yards'].describe())
two = ytg[ytg['TE'] == 2]
print(two['Yards'].describe())
three = ytg[ytg['TE'] == 3]
print(three['Yards'].describe())
four = ytg[ytg['TE'] == 4]
print(four['Yards'].describe())


# In[28]:


plt.hist(zero['Yards'])
plt.xlabel('Yards')
plt.ylabel('Times Occured')
plt.title('Zero TightEnds')
plt.savefig('C://Users//tilleymusprime//Desktop//ZeroTE.png')
plt.show()


# In[29]:


plt.hist(one['Yards'])
plt.xlabel('Yards')
plt.ylabel('Times Occured')
plt.title('One TightEnd')
plt.savefig('C://Users//tilleymusprime//Desktop//1TE.png')
plt.show()


# In[30]:


plt.hist(two['Yards'])
plt.xlabel('Yards')
plt.ylabel('Times Occured')
plt.title('Two TightEnds')
plt.savefig('C://Users//tilleymusprime//Desktop//2TE.png')
plt.show()


# In[31]:


plt.hist(three['Yards'])
plt.xlabel('Yards')
plt.ylabel('Times Occured')
plt.title('Three TightEnds')
plt.savefig('C://Users//tilleymusprime//Desktop//3TE.png')
plt.show()


# In[32]:


plt.hist(four['Yards'])
plt.xlabel('Yards')
plt.ylabel('Times Occured')
plt.title('Four TightEnds')
plt.savefig('C://Users//tilleymusprime//Desktop//FourTE.png')
plt.show()


# In[33]:


teamforma = df.groupby(['defense', 'DefensePersonnel'])['Yards'].mean()
teamforma.to_csv('C://Users//tilleymusprime//Desktop//teamdefenseformations.csv')


# In[34]:


#Next, we will take a look at defensive ranks within the defense formations and compare that with the overall defensive rank
three35 = df[df['DefensePersonnel'] == '3 DL, 3 LB, 5 DB']
team335 = three35.groupby('defense')['Yards'].mean()
team335 = pd.DataFrame(team335.sort_values())
team335['Rank'] = range(len(team335))

two45 = df[df['DefensePersonnel'] == '2 DL, 4 LB, 5 DB']
team245 = two45.groupby('defense')['Yards'].mean()
team245 = pd.DataFrame(team245.sort_values())
team245['Rank'] = range(len(team245))

three44 = df[df['DefensePersonnel'] == '3 DL, 4 LB, 4 DB']
team344 = three44.groupby('defense')['Yards'].mean()
team344 = pd.DataFrame(team344.sort_values())
team344['Rank'] = range(len(team344))

four25 = df[df['DefensePersonnel'] == '4 DL, 2 LB, 5 DB']                        
team425 = four25.groupby('defense')['Yards'].mean()
team425 = pd.DataFrame(team425.sort_values())
team425['Rank'] = range(len(team425))
                        
four34 = df[df['DefensePersonnel'] == '4 DL, 3 LB, 4 DB']                        
team434 = four34.groupby('defense')['Yards'].mean()
team434 = pd.DataFrame(team434.sort_values())
team434['Rank'] = range(len(team434))


teamtotal = df.groupby('defense')['Yards'].mean()
teamtotal = pd.DataFrame(teamtotal.sort_values())
teamtotal['Rank'] = range(len(teamtotal))


# In[35]:


a = pd.merge(team335, team245, how='outer', left_index=True, right_index=True)
a = pd.merge(a,team344, how='outer', left_index=True, right_index=True)
a = pd.merge(a, team425, how='outer', left_index=True, right_index=True)
a = pd.merge(a, team434,how='outer', left_index=True, right_index=True)
a = pd.merge(a, teamtotal,how='outer', left_index=True, right_index=True)
a.columns = ['335', '335Rank', '245', '245Rank', '344', '344Rank', '425', '425Rank', '434', '434Rank', 'total', 'totalrank']
a.sort_values('totalrank', inplace=True)


# In[ ]:





# In[57]:


#Next, we will look at the overall offensive ranks and compare that with rankings by each formation
off113 = df[df['OffensePersonnel'] == '1 RB, 1 TE, 3 WR']
team113 = off113.groupby('offense')['Yards'].mean()
team113 = pd.DataFrame(team113.sort_values(ascending=False))
team113['Rank'] = range(len(team113))

off122 = df[df['OffensePersonnel'] == '1 RB, 2 TE, 2 WR']
team122 = off122.groupby('offense')['Yards'].mean()
team122 = pd.DataFrame(team122.sort_values(ascending=False))
team122['Rank'] = range(len(team122))

off212 = df[df['OffensePersonnel'] == '2 RB, 1 TE, 2 WR']
team212 = off212.groupby('offense')['Yards'].mean()
team212 = pd.DataFrame(team212.sort_values(ascending=False))
team212['Rank'] = range(len(team212))

off131 = df[df['OffensePersonnel'] == '1 RB, 3 TE, 1 WR']
team131 = off131.groupby('offense')['Yards'].mean()
team131 = pd.DataFrame(team131.sort_values(ascending=False))
team131['Rank'] = range(len(team131))

off221 = df[df['OffensePersonnel'] == '2 RB, 2 TE, 1 WR']
team221 = off221.groupby('offense')['Yards'].mean()
team221 = pd.DataFrame(team221.sort_values(ascending=False))
team221['Rank'] = range(len(team221))

offtotal = df.groupby('offense')['Yards'].mean()
tototal = pd.DataFrame(offtotal.sort_values(ascending=False))
tototal['Rank'] = range(len(tototal))


# In[58]:


b = pd.merge(team113, team122, how='outer', left_index=True, right_index=True)
b = pd.merge(b, team212, how='outer', left_index=True, right_index=True)
b = pd.merge(b, team131, how='outer', left_index=True, right_index=True)
b = pd.merge(b, team221, how='outer', left_index=True, right_index=True)
b = pd.merge(b, tototal, how='outer', left_index=True, right_index=True)
b.columns = ['113', '113Rank', '122', '122Rank', '212', '212Rank', '131', '131Rank', '221', '221Rank','Total', 'TotalRank']
b = b.sort_values('TotalRank')
b
#It looks like 7 of the top 10 teams are also in the top 10 running with 3 wide receivers
#It also looks like the teams that are the worst at running the ball out of 3 receivers sets aaverage the fewest yards per rush


# In[64]:


print('Correlation113: ', pearsonr(b['TotalRank'], b['113Rank']))
print('Correlation122: ', pearsonr(b['TotalRank'], b['122Rank']))
print('Correlation131: ', pearsonr(b['TotalRank'], b['131Rank']))
#It looks like running with 3 receivers and 1 tight end is highly correlated with yards
#Sucess in 3 tight end formations seems to be unrelated to overall running success
plt.scatter(b['TotalRank'], b['113Rank'])
plt.xlabel('Rushing Rank')
plt.ylabel('Rank in 1 RB, 1 TE, 3 WR')
plt.savefig('C://Users//tilleymusprime//Desktop//3wr.png')
plt.show()


# In[65]:


plt.scatter(b['TotalRank'], b['131Rank'])
plt.xlabel('Rushing Rank')
plt.ylabel('Rank in 1 RB, 3 TE, 1 WR')
plt.savefig('C://Users//tilleymusprime//Desktop//3te.png')
plt.show()


# In[39]:


#Now we will combine the offensive and defensive rankings to see if being good at one means being good at the other
c = pd.merge(a,b,how='outer', left_index=True, right_index=True)
c = c[['TotalRank', 'totalrank']]
c.columns = ['Offense Rank', 'Defense Rank']
print(pearsonr(c['Offense Rank'], c['Defense Rank']))
#c.to_csv('C://Users//tilleymusprime//Desktop//odrank.csv')


# In[41]:


plt.scatter(c['Offense Rank'], c['Defense Rank'])
plt.xlabel('Offense Rank')
plt.ylabel('Defense Rank')
plt.title('Offensive Rank vs Defensive Rank')
plt.savefig('C://Users//tilleymusprime//Desktop//ovd.png')
plt.show()


# In[52]:


#Finally, lets see if rushing a lot leads to a higher average number of yards per play.
ovc = df['offense'].value_counts()
df2 = pd.merge(ovc, b, how='outer', left_index=True, right_index=True)

#It looks like if a team is good at running the ball, then they will run the ball more.
#If a team is not good at running the ball, running the ball more will not make them better.


# In[53]:


df2


# In[ ]:





# In[56]:


plt.scatter(df2['TotalRank'], df2['offense'])
plt.xlabel('Rush Ranking')
plt.ylabel('Number of runs')
plt.title('Offensive Rank vs Rushing Attempts')
plt.savefig('C://Users//tilleymusprime//Desktop//attemptsvsrank.png')
plt.show()


# In[48]:





# In[49]:





# In[47]:


pearsonr(df2['offense'], df2['TotalRank'])


# In[ ]:




