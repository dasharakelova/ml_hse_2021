#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# Task 1

# In[2]:


class Tree:
  '''Create a binary tree; keyword-only arguments `data`, `left`, `right`.

  Examples:
    l1 = Tree.leaf("leaf1")
    l2 = Tree.leaf("leaf2")
    tree = Tree(data="root", left=l1, right=Tree(right=l2))
  '''

  def leaf(data):
    '''Create a leaf tree
    '''
    return Tree(data=data)

  # pretty-print trees
  def __repr__(self):
    if self.is_leaf():
      return "Leaf(%r)" % self.data
    else:
      return "Tree(%r) { left = %r, right = %r }" % (self.data, self.left, self.right) 

  # all arguments after `*` are *keyword-only*!
  def __init__(self, *, data = None, left = None, right = None):
    self.data = data
    self.left = left
    self.right = right

  def is_leaf(self):
    '''Check if this tree is a leaf tree
    '''
    return self.left == None and self.right == None

  def children(self):
    '''List of child subtrees
    '''
    return [x for x in [self.left, self.right] if x]

  def depth(self):
    '''Compute the depth of a tree
    A leaf is depth-1, and a child is one deeper than the parent.
    '''
    return max([x.depth() for x in self.children()], default=0) + 1


# In[3]:


left_1_r = Tree.leaf('like')
right_1_r = Tree.leaf('nah')
Level_1_r = Tree(data='likedOtherSys?', left=left_1_r, right=right_1_r)
left_1_l = Tree.leaf('like')
right_1_l = Tree.leaf('nah')
Level_1_l = Tree(data='likedOtherSys?', left=left_1_l, right=right_1_l)
Level_2 = Tree(data='takenOtherSys?', left=Level_1_l, right=Level_1_r)
left_3 = Tree.leaf('like')
FinalTree = Tree(data='isSystems?', left=left_3, right=Level_2)


# In[4]:


print(FinalTree)


# Task 2

# In[5]:


df = pd.DataFrame([l.split(',') for l in '''rating,easy,ai,systems,theory,morning
2,True,True,False,True,False
2,True,True,False,True,False
2,False,True,False,False,False
2,False,False,False,True,False
2,False,True,True,False,True
1,True,True,False,False,False
1,True,True,False,True,False
1,False,True,False,True,False
0,False,False,False,False,True
0,True,False,False,True,True
0,False,True,False,True,False
0,True,True,True,True,True
-1,True,True,True,False,True
-1,False,False,True,True,False
-1,False,False,True,False,True
-1,True,False,True,False,True
-2,False,False,True,True,False
-2,False,True,True,False,True
-2,True,False,True,False,False
-2,True,False,True,False,True'''.splitlines()])


# In[6]:


new_header = df.iloc[0]
df = df[1:]
df.columns = new_header


# In[7]:


df['rating'] = pd.to_numeric(df['rating'])


# In[8]:


df['ok'] = np.where(df['rating'] >= 0, "True", "False")


# Task 3

# In[10]:


def single_feature_score(data, goal, feature):
    return np.mean([max(data[data[feature] == 'True'][goal].value_counts(normalize=True)), max(data[data[feature] == 'False'][goal].value_counts(normalize=True))])


# In[11]:


def best_feature(data, goal, features):
    return max(features, key=lambda f: single_feature_score(data, goal, f))


# In[12]:


def worst_feature(data, goal, features):
    return min(features, key=lambda f: single_feature_score(data, goal, f))


# In[13]:


print(best_feature(df, 'ok', ['ai', 'systems', 'morning', 'easy', 'theory']))
print(worst_feature(df, 'ok', ['ai', 'systems', 'morning', 'easy', 'theory']))


# Task 4

# In[14]:


def DecisionTreeTrain(data, goal, features, maxdepth='no'):
    while True:
        guess = data[goal].value_counts().idxmax()
        if data[goal].nunique() == 1:
            return Tree.leaf(guess)
        elif len(features) == 0:
            return Tree.leaf(guess)
        else:
            f = best_feature(data, goal, features)
            no = data[data[f] == 'False']
            yes = data[data[f] == 'True']
            features.remove(f)
            if isinstance(maxdepth, int):
                maxdepth = maxdepth - 1
            left = DecisionTreeTrain(no, goal, features, maxdepth=maxdepth)
            right = DecisionTreeTrain(yes, goal, features, maxdepth=maxdepth)
            tree = Tree(data=f, left=left, right=right)
            if isinstance(maxdepth, int) and tree.depth() > (maxdepth + 2):
                return Tree.leaf(guess)
                break
            return tree


# In[15]:


DecisionTreeTrain(df, 'ok', ['ai', 'systems', 'morning', 'easy', 'theory'])


# In[16]:


def DecisionTreeTest(tree, test, i):
    if tree.is_leaf():
        return tree
    else:
        if test.iloc[i][tree.data] == 'False':
            return DecisionTreeTest(tree.left, test, i)
        elif test.iloc[i][tree.data] == 'True':
            return DecisionTreeTest(tree.right, test, i)


# Now we'll write a function that computes the score of a given tree by its performance on given data by guessing the label of every row in the data.

# In[17]:


def dataset_score(tree, data, goal):
    correct = 0
    for i in range(len(data)):
        if data.iloc[i][goal] == DecisionTreeTest(tree, data, i).data:
            correct +=1
    return correct/len(data)


# In[18]:


dataset_score(DecisionTreeTrain(df, 'ok', ['ai', 'systems', 'morning', 'easy', 'theory']), df, 'ok')


# Now we'll train five trees of different depths and compute the score for each of them.

# In[28]:


scores = []
for i in range(1,6):
    scores.append(dataset_score(DecisionTreeTrain(df, 'ok', ['ai', 'systems', 'morning', 'easy', 'theory'], maxdepth=i), df, 'ok'))


# In[33]:


# plt.plot([i for i in range(1,6)], scores)
# plt.xlabel('Tree depth')
# plt.ylabel('Score')
# plt.title('Performance of trees with different maxdepth values')
# plt.savefig('trees_performance.png')

